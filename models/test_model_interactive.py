"""
Interactive Model Testing Script

This script allows you to manually test the trained position-aware model
by showing initial state, move, expected state, and predicted state in a 
clear, visual format.
"""

import torch
import torch.nn as nn
import sys
import os
import numpy as np
from torch.utils.data import DataLoader
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import PreTrainDataset, VOCAB, COLOR_MAP
from models.general_model import PositionAwareFoundationalModel
from data_generation.cube_state_gen import generate_random_scramble
from cuber.cuber import Cube

def format_cube_state(state_tensor, title="Cube State"):
    """
    Format a cube state tensor into a readable 3D cube representation.
    
    Args:
        state_tensor: torch.Tensor of shape (54,) with color indices
        title: String title for the display
    
    Returns:
        String representation of the cube
    """
    # Convert tensor to color names
    colors = []
    for idx in state_tensor:
        color_idx = idx.item()
        for token, vocab_idx in VOCAB.items():
            if vocab_idx == color_idx:
                colors.append(token)
                break
    
    # Cube layout (54 stickers):
    # Faces: U(0-8), L(9-17), F(18-26), R(27-35), B(36-44), D(45-53)
    
    output = []
    output.append(f"\n{title}:")
    output.append("="*50)
    
    # Top face (U: 0-8)
    output.append("\n    TOP FACE (U)")
    output.append(f"    {colors[0]:6s} {colors[1]:6s} {colors[2]:6s}")
    output.append(f"    {colors[3]:6s} {colors[4]:6s} {colors[5]:6s}")
    output.append(f"    {colors[6]:6s} {colors[7]:6s} {colors[8]:6s}")
    
    # Middle band: L, F, R, B faces side by side
    output.append("\n   LEFT    FRONT   RIGHT    BACK")
    for row in range(3):
        l_start = 9 + row * 3
        f_start = 18 + row * 3
        r_start = 27 + row * 3
        b_start = 36 + row * 3
        
        l_colors = f"{colors[l_start]:6s} {colors[l_start+1]:6s} {colors[l_start+2]:6s}"
        f_colors = f"{colors[f_start]:6s} {colors[f_start+1]:6s} {colors[f_start+2]:6s}"
        r_colors = f"{colors[r_start]:6s} {colors[r_start+1]:6s} {colors[r_start+2]:6s}"
        b_colors = f"{colors[b_start]:6s} {colors[b_start+1]:6s} {colors[b_start+2]:6s}"
        
        output.append(f"   {l_colors} {f_colors} {r_colors} {b_colors}")
    
    # Bottom face (D: 45-53)
    output.append("\n    BOTTOM FACE (D)")
    output.append(f"    {colors[45]:6s} {colors[46]:6s} {colors[47]:6s}")
    output.append(f"    {colors[48]:6s} {colors[49]:6s} {colors[50]:6s}")
    output.append(f"    {colors[51]:6s} {colors[52]:6s} {colors[53]:6s}")
    
    return "\n".join(output)

def get_move_name(move_tensor):
    """Convert move tensor to move name."""
    move_idx = move_tensor.item()
    for token, idx in VOCAB.items():
        if idx == move_idx:
            return token
    return "UNKNOWN"

def compare_states(expected, predicted, title="State Comparison"):
    """Compare expected vs predicted states and show differences."""
    output = []
    output.append(f"\n{title}:")
    output.append("="*50)
    
    correct = 0
    total = 54
    differences = []
    
    for i in range(54):
        exp_idx = expected[i].item()
        pred_idx = predicted[i].item()
        
        # Get color names
        exp_color = "UNK"
        pred_color = "UNK"
        for token, idx in VOCAB.items():
            if idx == exp_idx:
                exp_color = token
            if idx == pred_idx:
                pred_color = token
        
        if exp_idx == pred_idx:
            correct += 1
        else:
            differences.append({
                'position': i,
                'expected': exp_color,
                'predicted': pred_color,
                'face': get_face_name(i)
            })
    
    accuracy = correct / total
    output.append(f"Accuracy: {accuracy:.2%} ({correct}/{total} correct)")
    
    if differences:
        output.append(f"\nDifferences found:")
        for diff in differences:
            output.append(f"  Position {diff['position']:2d} ({diff['face']}): "
                         f"Expected {diff['expected']:6s}, Got {diff['predicted']:6s}")
    else:
        output.append("\nPERFECT MATCH! All stickers predicted correctly!")
    
    return "\n".join(output)

def get_face_name(position):
    """Get face name for a sticker position."""
    if 0 <= position <= 8:
        return f"U{position}"
    elif 9 <= position <= 17:
        return f"L{position-9}"
    elif 18 <= position <= 26:
        return f"F{position-18}"
    elif 27 <= position <= 35:
        return f"R{position-27}"
    elif 36 <= position <= 44:
        return f"B{position-36}"
    elif 45 <= position <= 53:
        return f"D{position-45}"
    else:
        return "?"

def test_model_interactive(num_tests=5, use_saved_model=True):
    """
    Interactive model testing with visual cube state display.
    
    Args:
        num_tests: Number of test cases to show
        use_saved_model: Whether to load the best saved model
    """
    print("INTERACTIVE MODEL TESTING")
    print("="*60)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
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
    
    if use_saved_model and os.path.exists("checkpoints/best_model.pth"):
        print("Loading best saved model...")
        model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
        print("Model loaded successfully!")
    else:
        print("WARNING: Using randomly initialized model (no saved model found)")
    
    model.eval()
    
    # Create test dataset
    print(f"\nCreating test dataset...")
    test_dataset = PreTrainDataset(100, seed=999)  # Small test set
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Color indices for proper prediction
    color_indices = list(range(len(VOCAB) - 6, len(VOCAB)))
    
    print(f"\nRunning {num_tests} interactive tests...")
    
    with torch.no_grad():
        for test_num, (initial_state, move, expected_state) in enumerate(test_dataloader):
            if test_num >= num_tests:
                break
            
            print(f"\n" + "="*80)
            print(f"TEST CASE #{test_num + 1}")
            print("="*80)
            
            # Move to device
            initial_state = initial_state.to(device)
            move = move.to(device)
            expected_state = expected_state.to(device)
            
            # Get move name
            move_name = get_move_name(move[0])
            print(f"\nMOVE: {move_name}")
            
            # Display initial state
            print(format_cube_state(initial_state[0], "INITIAL STATE"))
            
            # Get model prediction
            output_logits = model(initial_state, move)
            color_logits = output_logits[:, :, color_indices]
            predicted_state = torch.argmax(color_logits, dim=-1) + color_indices[0]
            
            # Display expected state
            print(format_cube_state(expected_state[0], "EXPECTED STATE (after move)"))
            
            # Display predicted state
            print(format_cube_state(predicted_state[0], "PREDICTED STATE (model output)"))
            
            # Compare states
            print(compare_states(expected_state[0], predicted_state[0], "COMPARISON"))
            
            # Calculate prediction confidence
            probabilities = torch.softmax(color_logits[0], dim=-1)
            confidence_scores = torch.max(probabilities, dim=-1)[0]
            avg_confidence = confidence_scores.mean().item()
            min_confidence = confidence_scores.min().item()
            
            print(f"\nPREDICTION CONFIDENCE:")
            print(f"   Average: {avg_confidence:.1%}")
            print(f"   Minimum: {min_confidence:.1%}")
            
            # Show low confidence positions
            low_conf_threshold = 0.8
            low_conf_positions = (confidence_scores < low_conf_threshold).nonzero().flatten()
            if len(low_conf_positions) > 0:
                print(f"\nLow confidence positions (< {low_conf_threshold:.0%}):")
                for pos in low_conf_positions:
                    face = get_face_name(pos.item())
                    conf = confidence_scores[pos].item()
                    print(f"   Position {pos.item():2d} ({face}): {conf:.1%}")
            else:
                print(f"\nAll predictions have high confidence!")
            
            # Wait for user input to continue (except for last test)
            if test_num < num_tests - 1:
                input(f"\nPress Enter to see next test case...")
    
    print(f"\nInteractive testing completed!")
    print(f"TIP: You can modify this script to test specific cube states or moves.")

def test_specific_move(move_name, scramble_length=5):
    """
    Test a specific move with a custom scramble.
    
    Args:
        move_name: String name of the move (e.g., "R", "U'", "x2")
        scramble_length: Length of initial scramble
    """
    print(f"TESTING SPECIFIC MOVE: {move_name}")
    print("="*50)
    
    # Validate move
    if move_name not in VOCAB:
        print(f"ERROR: Move '{move_name}' not found in vocabulary!")
        print(f"Valid moves: {', '.join([m for m in VOCAB.keys() if m not in ['WHITE', 'YELLOW', 'GREEN', 'BLUE', 'RED', 'ORANGE']])}")
        return
    
    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = PositionAwareFoundationalModel(
        vocab_size=len(VOCAB),
        d_model=256,
        nhead=8,
        d_hid=1024,
        n_encoder_layers=4,
        n_decoder_layers=2,
        dropout=0.1
    ).to(device)
    
    if os.path.exists("checkpoints/best_model.pth"):
        model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
        print("Model loaded successfully!")
    else:
        print("WARNING: No saved model found!")
        return
    
    model.eval()

    # Create scrambled cube
    scramble = generate_random_scramble(scramble_length)
    cube = Cube()
    cube.turn(scramble)
    initial_state = cube.get_faces_str("ULFRBD")
    
    # Apply the specific move
    cube.turn(move_name)
    expected_state = cube.get_faces_str("ULFRBD")
    
    # Convert to tensors
    initial_tensor = torch.tensor([VOCAB[COLOR_MAP[face]] for face in initial_state.split(" ")], dtype=torch.long).unsqueeze(0).to(device)
    move_tensor = torch.tensor([VOCAB[move_name]], dtype=torch.long).to(device)
    expected_tensor = torch.tensor([VOCAB[COLOR_MAP[face]] for face in expected_state.split(" ")], dtype=torch.long).unsqueeze(0).to(device)
    
    print(f"Initial scramble: {scramble}")
    print(f"Testing move: {move_name}")
    
    # Get prediction
    color_indices = list(range(len(VOCAB) - 6, len(VOCAB)))
    
    with torch.no_grad():
        output_logits = model(initial_tensor, move_tensor)
        color_logits = output_logits[:, :, color_indices]
        predicted_tensor = torch.argmax(color_logits, dim=-1) + color_indices[0]
    
    # Display results
    print(format_cube_state(initial_tensor[0], "INITIAL STATE"))
    print(format_cube_state(expected_tensor[0], "EXPECTED STATE"))
    print(format_cube_state(predicted_tensor[0], "PREDICTED STATE"))
    print(compare_states(expected_tensor[0], predicted_tensor[0]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Model Testing")
    parser.add_argument("--num_tests", type=int, default=5, help="Number of test cases to run")
    parser.add_argument("--specific_move", type=str, help="Test a specific move (e.g., 'R', 'U\\'', 'x2')")
    parser.add_argument("--scramble_length", type=int, default=5, help="Scramble length for specific move test")
    
    args = parser.parse_args()
    
    if args.specific_move:
        test_specific_move(args.specific_move, args.scramble_length)
    else:
        test_model_interactive(args.num_tests)
