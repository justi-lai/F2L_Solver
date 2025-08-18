"""
Test the model's performance on multi-move sequences.

This script evaluates how well the current single-move trained model
performs when given sequences of multiple moves to predict the final state.
"""

import torch
import torch.nn as nn
import os
import sys
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import VOCAB, COLOR_MAP
from models.general_model import PositionAwareFoundationalModel
from data_generation.cube_state_gen import generate_random_scramble

def generate_multi_move_data(num_samples, sequence_lengths=[2, 3, 4, 5], seed=42):
    """
    Generate test data with multiple move sequences.
    
    Args:
        num_samples: Total number of samples to generate
        sequence_lengths: List of sequence lengths to test
        seed: Random seed for reproducibility
    
    Returns:
        List of (initial_state, move_sequence, final_state) tuples
    """
    from cuber.cuber import Cube
    
    np.random.seed(seed)
    data = []
    
    samples_per_length = num_samples // len(sequence_lengths)
    
    print(f"Generating {num_samples} multi-move samples...")
    
    with tqdm(total=num_samples, desc="Generating multi-move data") as pbar:
        for seq_len in sequence_lengths:
            for _ in range(samples_per_length):
                # Generate initial scramble
                initial_scramble = generate_random_scramble(np.random.randint(5, 15))
                
                # Create cube and apply initial scramble
                cube = Cube()
                cube.turn(initial_scramble)
                initial_state = cube.get_faces_str("ULFRBD")
                
                # Generate sequence of moves
                all_moves = list(VOCAB.keys())
                move_tokens = [token for token in all_moves if token not in ['WHITE', 'YELLOW', 'GREEN', 'BLUE', 'RED', 'ORANGE']]
                
                move_sequence = []
                for _ in range(seq_len):
                    move = np.random.choice(move_tokens)
                    move_sequence.append(move)
                    cube.turn(move)
                
                final_state = cube.get_faces_str("ULFRBD")
                
                data.append((initial_state, move_sequence, final_state))
                pbar.update(1)
    
    return data

def evaluate_multi_move_model(model, test_data, device, max_samples=None):
    """
    Evaluate model on multi-move sequences.
    
    Args:
        model: The trained model
        test_data: List of (initial_state, move_sequence, final_state) tuples
        device: Device to run on
        max_samples: Maximum number of samples to evaluate (None for all)
    
    Returns:
        dict: Evaluation results
    """
    model.eval()
    
    # Color indices for proper evaluation
    color_indices = list(range(len(VOCAB) - 6, len(VOCAB)))
    
    results_by_length = {}
    
    if max_samples:
        test_data = test_data[:max_samples]
    
    print(f"Evaluating on {len(test_data)} multi-move samples...")
    
    with torch.no_grad():
        for initial_state_str, move_sequence, final_state_str in tqdm(test_data, desc="Evaluating"):
            seq_len = len(move_sequence)
            
            if seq_len not in results_by_length:
                results_by_length[seq_len] = {
                    'correct_predictions': 0,
                    'total_predictions': 0,
                    'perfect_predictions': 0,
                    'total_samples': 0
                }
            
            # Convert to tensors
            initial_tensor = torch.tensor([VOCAB[COLOR_MAP[face]] for face in initial_state_str.split(" ")], dtype=torch.long).unsqueeze(0).to(device)
            move_tensors = torch.tensor([VOCAB[move] for move in move_sequence], dtype=torch.long).unsqueeze(0).to(device)
            final_tensor = torch.tensor([VOCAB[COLOR_MAP[face]] for face in final_state_str.split(" ")], dtype=torch.long).unsqueeze(0).to(device)
            
            # Test the model's approach to multi-move sequences
            # Method 1: Try using the sequence directly (if model supports it)
            try:
                if hasattr(model, 'forward_sequence'):
                    output_logits = model.forward_sequence(initial_tensor, move_tensors)
                else:
                    # Method 2: Apply moves iteratively (current state prediction)
                    current_state = initial_tensor
                    for move_idx in range(seq_len):
                        move = move_tensors[:, move_idx:move_idx+1]
                        output_logits = model(current_state, move.squeeze(-1))
                        
                        # Get predicted next state for next iteration
                        color_logits = output_logits[:, :, color_indices]
                        predicted_state = torch.argmax(color_logits, dim=-1) + color_indices[0]
                        current_state = predicted_state
                    
                    # Final output_logits is already set from the last iteration
                
                # Extract color predictions
                color_logits = output_logits[:, :, color_indices]
                final_state_colors = final_tensor - color_indices[0]
                
                # Calculate accuracy
                predictions = torch.argmax(color_logits, dim=-1)
                correct = (predictions == final_state_colors).sum().item()
                total = final_state_colors.numel()
                
                results_by_length[seq_len]['correct_predictions'] += correct
                results_by_length[seq_len]['total_predictions'] += total
                
                # Check for perfect prediction (all 54 stickers correct)
                if torch.all(predictions == final_state_colors):
                    results_by_length[seq_len]['perfect_predictions'] += 1
                
                results_by_length[seq_len]['total_samples'] += 1
                
            except Exception as e:
                print(f"Error processing sequence length {seq_len}: {e}")
                continue
    
    # Calculate final statistics
    overall_stats = {
        'by_length': {},
        'overall': {
            'accuracy': 0,
            'perfect_rate': 0,
            'total_samples': 0
        }
    }
    
    total_correct = 0
    total_predictions = 0
    total_perfect = 0
    total_samples = 0
    
    for seq_len, stats in results_by_length.items():
        if stats['total_samples'] > 0:
            accuracy = stats['correct_predictions'] / stats['total_predictions']
            perfect_rate = stats['perfect_predictions'] / stats['total_samples']
            
            overall_stats['by_length'][seq_len] = {
                'accuracy': accuracy,
                'perfect_rate': perfect_rate,
                'samples': stats['total_samples']
            }
            
            total_correct += stats['correct_predictions']
            total_predictions += stats['total_predictions']
            total_perfect += stats['perfect_predictions']
            total_samples += stats['total_samples']
    
    if total_predictions > 0:
        overall_stats['overall'] = {
            'accuracy': total_correct / total_predictions,
            'perfect_rate': total_perfect / total_samples,
            'total_samples': total_samples
        }
    
    return overall_stats

def main():
    """Main testing function."""
    print("MULTI-MOVE SEQUENCE TESTING")
    print("="*50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load the best model
    model = PositionAwareFoundationalModel(
        vocab_size=len(VOCAB),
        d_model=256,
        nhead=8,
        d_hid=1024,
        n_encoder_layers=4,
        n_decoder_layers=2,
        dropout=0.1
    ).to(device)
    
    model_path = "checkpoints/best_multi_move_model.pth"
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found!")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully!")
    
    # Generate test data for different sequence lengths
    test_data = generate_multi_move_data(
        num_samples=400,  # 100 samples per sequence length
        sequence_lengths=[2, 3, 4, 5],
        seed=54321
    )
    
    # Evaluate model performance
    results = evaluate_multi_move_model(model, test_data, device, max_samples=400)
    
    # Display results
    print(f"\nMULTI-MOVE EVALUATION RESULTS")
    print("="*50)
    
    print(f"PERFORMANCE BY SEQUENCE LENGTH:")
    print("-" * 40)
    
    for seq_len in sorted(results['by_length'].keys()):
        stats = results['by_length'][seq_len]
        print(f"Length {seq_len}: Accuracy: {stats['accuracy']:6.2%} | "
              f"Perfect: {stats['perfect_rate']:6.2%} | "
              f"Samples: {stats['samples']:3d}")
    
    print(f"\nOVERALL PERFORMANCE:")
    print("-" * 40)
    overall = results['overall']
    print(f"Total Accuracy: {overall['accuracy']:6.2%}")
    print(f"Perfect Rate:   {overall['perfect_rate']:6.2%}")
    print(f"Total Samples:  {overall['total_samples']:3d}")
    
    # Determine if fine-tuning is needed
    print(f"\nRECOMMENDATION:")
    print("-" * 40)
    
    if overall['accuracy'] < 0.8:  # Less than 80% accuracy
        print("NEEDS FINE-TUNING: Performance on multi-move sequences is poor.")
        print("Recommendation: Create multi-move dataset and fine-tune the model.")
        fine_tune_needed = True
    elif overall['perfect_rate'] < 0.5:  # Less than 50% perfect predictions
        print("COULD BENEFIT FROM FINE-TUNING: Accuracy is decent but perfect predictions are low.")
        print("Recommendation: Fine-tuning could improve multi-move performance.")
        fine_tune_needed = True
    else:
        print("PERFORMANCE IS GOOD: Model handles multi-move sequences well.")
        print("Fine-tuning may not be necessary.")
        fine_tune_needed = False
    
    return fine_tune_needed, results

if __name__ == "__main__":
    needs_tuning, results = main()
