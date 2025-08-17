"""
Automatic checkpoint evaluation script (no user input required).

This script evaluates all available checkpoints and automatically cleans up
poor performing ones, keeping only the best_model.pth and best epoch checkpoint.
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

def evaluate_checkpoint(model_path, test_dataloader, device):
    """Evaluate a single checkpoint on the test dataset."""
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
    
    with torch.no_grad():
        for initial_state, move, final_state in test_dataloader:
            initial_state = initial_state.to(device)
            move = move.to(device)
            final_state = final_state.to(device)
            
            # Forward pass
            output_logits = model(initial_state, move)
            color_logits = output_logits[:, :, color_indices]
            final_state_colors = final_state - color_indices[0]
            
            # Calculate loss and accuracy
            batch_loss = loss_fn(color_logits.view(-1, len(color_indices)), final_state_colors.view(-1))
            total_loss += batch_loss.item()
            
            predictions = torch.argmax(color_logits, dim=-1)
            batch_correct = (predictions == final_state_colors).sum().item()
            correct_predictions += batch_correct
            total_predictions += final_state_colors.numel()
            
            # Check for perfect predictions
            batch_size = final_state_colors.shape[0]
            for i in range(batch_size):
                if torch.all(predictions[i] == final_state_colors[i]):
                    perfect_predictions += 1
            
            num_batches += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    perfect_rate = perfect_predictions / len(test_dataloader.dataset) if len(test_dataloader.dataset) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'perfect_predictions': perfect_predictions,
        'perfect_rate': perfect_rate,
        'total_samples': len(test_dataloader.dataset),
        'error': None
    }

def main():
    """Main evaluation and cleanup function."""
    print("AUTOMATIC CHECKPOINT EVALUATION AND CLEANUP")
    print("="*50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create test dataset
    test_size = 1000  # Sufficient for reliable comparison
    print(f"Creating test dataset ({test_size:,} samples)...")
    
    test_dataset = PreTrainDataset(test_size, seed=12345)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Find checkpoint files
    checkpoint_dir = "checkpoints"
    checkpoint_files = []
    
    if os.path.exists(checkpoint_dir):
        for filename in os.listdir(checkpoint_dir):
            if filename.endswith('.pth'):
                checkpoint_files.append(os.path.join(checkpoint_dir, filename))
    
    if not checkpoint_files:
        print("No checkpoint files found!")
        return
    
    checkpoint_files.sort()
    print(f"\nFound {len(checkpoint_files)} checkpoint files")
    
    # Evaluate each checkpoint
    results = []
    
    for checkpoint_path in checkpoint_files:
        checkpoint_name = os.path.basename(checkpoint_path)
        print(f"Evaluating {checkpoint_name}...", end=" ")
        
        result = evaluate_checkpoint(checkpoint_path, test_dataloader, device)
        result['name'] = checkpoint_name
        result['path'] = checkpoint_path
        results.append(result)
        
        if result['error']:
            print(f"ERROR: {result['error']}")
        else:
            print(f"Acc: {result['accuracy']:.2%}, Perfect: {result['perfect_rate']:.2%}")
    
    # Sort and analyze results
    valid_results = [r for r in results if not r['error']]
    valid_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    if not valid_results:
        print("No valid checkpoints found!")
        return
    
    print(f"\nRESULTS RANKING:")
    print("-" * 50)
    
    for i, result in enumerate(valid_results):
        rank_symbol = "★" if i == 0 else " "
        print(f"{rank_symbol} {i+1:2d}. {result['name']:<25} | "
              f"Acc: {result['accuracy']:6.2%} | "
              f"Perfect: {result['perfect_rate']:6.2%}")
    
    # Determine files to keep
    best_checkpoint = valid_results[0]
    print(f"\nBEST PERFORMING: {best_checkpoint['name']} ({best_checkpoint['accuracy']:.2%})")
    
    # Keep best_model.pth and best epoch checkpoint
    files_to_keep = []
    
    # Always keep best_model.pth if it exists
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        files_to_keep.append(best_model_path)
    
    # Find best epoch checkpoint
    epoch_checkpoints = [r for r in valid_results if 'epoch' in r['name']]
    if epoch_checkpoints:
        best_epoch_checkpoint = epoch_checkpoints[0]
        if best_epoch_checkpoint['path'] not in files_to_keep:
            files_to_keep.append(best_epoch_checkpoint['path'])
        print(f"BEST EPOCH CHECKPOINT: {best_epoch_checkpoint['name']} ({best_epoch_checkpoint['accuracy']:.2%})")
    
    # Determine files to remove
    files_to_remove = []
    total_size_mb = 0
    
    for result in valid_results:
        if result['path'] not in files_to_keep:
            files_to_remove.append(result['path'])
            file_size = os.path.getsize(result['path']) / (1024*1024)
            total_size_mb += file_size
    
    # Show cleanup plan
    print(f"\nCLEANUP PLAN:")
    print(f"KEEPING {len(files_to_keep)} files:")
    for file_path in files_to_keep:
        file_size = os.path.getsize(file_path) / (1024*1024)
        print(f"  ✓ {os.path.basename(file_path)} ({file_size:.1f} MB)")
    
    if files_to_remove:
        print(f"\nREMOVING {len(files_to_remove)} files ({total_size_mb:.1f} MB):")
        deleted_count = 0
        for file_path in files_to_remove:
            file_size = os.path.getsize(file_path) / (1024*1024)
            try:
                os.remove(file_path)
                print(f"  ✓ Deleted {os.path.basename(file_path)} ({file_size:.1f} MB)")
                deleted_count += 1
            except Exception as e:
                print(f"  ✗ Failed to delete {os.path.basename(file_path)}: {e}")
        
        print(f"\nCLEANUP COMPLETE!")
        print(f"  - Deleted: {deleted_count}/{len(files_to_remove)} files")
        print(f"  - Space saved: {total_size_mb:.1f} MB")
        print(f"  - Kept best performing checkpoints only")
    else:
        print(f"\nNo files to remove - all checkpoints are optimal.")
    
    print(f"\nFINAL STATUS:")
    remaining_files = [f for f in checkpoint_files if os.path.exists(f)]
    print(f"  - Checkpoints remaining: {len(remaining_files)}")
    for file_path in remaining_files:
        file_size = os.path.getsize(file_path) / (1024*1024)
        print(f"    • {os.path.basename(file_path)} ({file_size:.1f} MB)")

if __name__ == "__main__":
    main()
