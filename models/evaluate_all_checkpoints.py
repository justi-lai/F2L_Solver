"""
Comprehensive checkpoint evaluation script.

This script evaluates all available checkpoints on a large test dataset
to determine which ones to keep based on performance.
"""

import torch
import torch.nn as nn
import os
import sys
from torch.utils.data import DataLoader
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import PreTrainDataset, VOCAB
from models.general_model import PositionAwareFoundationalModel

def evaluate_checkpoint(model_path, test_dataloader, device, verbose=False):
    """
    Evaluate a single checkpoint on the test dataset.
    
    Args:
        model_path: Path to the checkpoint file
        test_dataloader: DataLoader for test data
        device: Device to run evaluation on
        verbose: Whether to print detailed results
    
    Returns:
        dict: Evaluation results
    """
    # Load model
    model = PositionAwareFoundationalModel(
        vocab_size=len(VOCAB),
        d_model=256,
        nhead=8,
        d_hid=1024,
        n_encoder_layers=4,
        n_decoder_layers=2,
        dropout=0.1
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        return {
            'error': str(e),
            'accuracy': 0.0,
            'loss': float('inf'),
            'perfect_predictions': 0,
            'total_samples': 0
        }
    
    model.eval()
    
    # Color indices for proper evaluation
    color_indices = list(range(len(VOCAB) - 6, len(VOCAB)))
    loss_fn = nn.CrossEntropyLoss()
    
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    perfect_predictions = 0
    num_batches = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (initial_state, move, final_state) in enumerate(test_dataloader):
            initial_state = initial_state.to(device)
            move = move.to(device)
            final_state = final_state.to(device)
            
            # Forward pass
            output_logits = model(initial_state, move)
            
            # Extract only color logits (last 6 dimensions)
            color_logits = output_logits[:, :, color_indices]
            
            # Adjust targets for color-only prediction
            final_state_colors = final_state - color_indices[0]
            
            # Calculate loss
            batch_loss = loss_fn(color_logits.view(-1, len(color_indices)), final_state_colors.view(-1))
            total_loss += batch_loss.item()
            
            # Calculate accuracy
            predictions = torch.argmax(color_logits, dim=-1)
            batch_correct = (predictions == final_state_colors).sum().item()
            correct_predictions += batch_correct
            total_predictions += final_state_colors.numel()
            
            # Check for perfect batch predictions (all 54 stickers correct)
            batch_size = final_state_colors.shape[0]
            for i in range(batch_size):
                if torch.all(predictions[i] == final_state_colors[i]):
                    perfect_predictions += 1
            
            num_batches += 1
            
            if verbose and batch_idx % 50 == 0:
                current_acc = correct_predictions / total_predictions
                print(f"  Batch {batch_idx:3d}: {current_acc:.2%} accuracy")
    
    eval_time = time.time() - start_time
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    perfect_rate = perfect_predictions / len(test_dataloader.dataset) if len(test_dataloader.dataset) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'perfect_predictions': perfect_predictions,
        'perfect_rate': perfect_rate,
        'total_samples': len(test_dataloader.dataset),
        'eval_time': eval_time,
        'error': None
    }

def main():
    """Main evaluation function."""
    print("COMPREHENSIVE CHECKPOINT EVALUATION")
    print("="*60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create a large test dataset for fair evaluation
    test_size = 2000  # Large enough for reliable comparison
    print(f"Creating test dataset with {test_size:,} samples...")
    
    test_dataset = PreTrainDataset(test_size, seed=12345)  # Different seed from training
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Test dataset created successfully.\n")
    
    # Find all checkpoint files
    checkpoint_dir = "checkpoints"
    checkpoint_files = []
    
    if os.path.exists(checkpoint_dir):
        for filename in os.listdir(checkpoint_dir):
            if filename.endswith('.pth'):
                checkpoint_files.append(os.path.join(checkpoint_dir, filename))
    
    if not checkpoint_files:
        print("No checkpoint files found!")
        return
    
    # Sort checkpoint files for consistent ordering
    checkpoint_files.sort()
    
    print(f"Found {len(checkpoint_files)} checkpoint files:")
    for i, cp in enumerate(checkpoint_files):
        print(f"  {i+1}. {os.path.basename(cp)}")
    print()
    
    # Evaluate each checkpoint
    results = []
    
    for i, checkpoint_path in enumerate(checkpoint_files):
        checkpoint_name = os.path.basename(checkpoint_path)
        print(f"Evaluating {checkpoint_name} ({i+1}/{len(checkpoint_files)})...")
        
        result = evaluate_checkpoint(checkpoint_path, test_dataloader, device, verbose=False)
        result['name'] = checkpoint_name
        result['path'] = checkpoint_path
        results.append(result)
        
        if result['error']:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Accuracy: {result['accuracy']:.2%}")
            print(f"  Perfect predictions: {result['perfect_predictions']}/{result['total_samples']} ({result['perfect_rate']:.2%})")
            print(f"  Loss: {result['loss']:.4f}")
            print(f"  Evaluation time: {result['eval_time']:.1f}s")
        print()
    
    # Sort results by accuracy (descending)
    valid_results = [r for r in results if not r['error']]
    valid_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    if not valid_results:
        print("No valid checkpoints found!")
        return
    
    # Display ranking
    print("CHECKPOINT RANKING (by accuracy):")
    print("="*60)
    
    for i, result in enumerate(valid_results):
        rank = i + 1
        print(f"{rank:2d}. {result['name']:<25} | "
              f"Acc: {result['accuracy']:6.2%} | "
              f"Perfect: {result['perfect_rate']:6.2%} | "
              f"Loss: {result['loss']:7.4f}")
    
    print()
    
    # Recommendations
    best_checkpoint = valid_results[0]
    print("RECOMMENDATIONS:")
    print("="*60)
    print(f"BEST CHECKPOINT: {best_checkpoint['name']}")
    print(f"  - Accuracy: {best_checkpoint['accuracy']:.2%}")
    print(f"  - Perfect predictions: {best_checkpoint['perfect_rate']:.2%}")
    print(f"  - Loss: {best_checkpoint['loss']:.4f}")
    print()
    
    # Find best performing checkpoint (excluding best_model.pth)
    epoch_checkpoints = [r for r in valid_results if 'epoch' in r['name']]
    
    if epoch_checkpoints:
        best_epoch_checkpoint = epoch_checkpoints[0]
        print(f"BEST EPOCH CHECKPOINT: {best_epoch_checkpoint['name']}")
        print(f"  - Accuracy: {best_epoch_checkpoint['accuracy']:.2%}")
        print(f"  - Perfect predictions: {best_epoch_checkpoint['perfect_rate']:.2%}")
        print(f"  - Loss: {best_epoch_checkpoint['loss']:.4f}")
        print()
    
    # Files to keep
    files_to_keep = [best_checkpoint['path']]
    if epoch_checkpoints and best_epoch_checkpoint['path'] != best_checkpoint['path']:
        files_to_keep.append(best_epoch_checkpoint['path'])
    
    # Files to remove
    files_to_remove = []
    for result in results:
        if result['path'] not in files_to_keep and not result['error']:
            files_to_remove.append(result['path'])
    
    print(f"RECOMMENDED ACTIONS:")
    print(f"KEEP ({len(files_to_keep)} files):")
    for file_path in files_to_keep:
        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
        print(f"  - {os.path.basename(file_path)} ({file_size:.1f} MB)")
    
    if files_to_remove:
        print(f"\nREMOVE ({len(files_to_remove)} files):")
        total_size_mb = 0
        for file_path in files_to_remove:
            file_size = os.path.getsize(file_path) / (1024*1024)  # MB
            total_size_mb += file_size
            print(f"  - {os.path.basename(file_path)} ({file_size:.1f} MB)")
        print(f"\nTotal space saved: {total_size_mb:.1f} MB")
        
        # Ask for confirmation and delete
        print(f"\nWould you like to delete the underperforming checkpoints? (y/n): ", end="")
        response = input().strip().lower()
        
        if response == 'y' or response == 'yes':
            deleted_count = 0
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                    print(f"Deleted: {os.path.basename(file_path)}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Failed to delete {os.path.basename(file_path)}: {e}")
            
            print(f"\nDeleted {deleted_count}/{len(files_to_remove)} files successfully!")
            print(f"Kept {len(files_to_keep)} best performing checkpoints.")
        else:
            print("No files deleted. You can manually delete them later if desired.")
    else:
        print(f"\nAll checkpoints are performing equally well - no files to remove.")

if __name__ == "__main__":
    main()
