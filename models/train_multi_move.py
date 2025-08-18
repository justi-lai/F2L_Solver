"""
Fine-tune the model for multi-move sequence prediction.

This script takes the trained single-move model and fine-tunes it
on multi-move sequences to improve its ability to handle longer sequences.
"""

import torch
import torch.nn as nn
import os
import sys
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import VOCAB
from datasets import CurriculumMultiMoveDataset, AlignedEvalMultiMoveDataset
from models.general_model import PositionAwareFoundationalModel

def evaluate_multi_model(model, dataloader, loss_fn, device, max_sequence_length):
    """
    Evaluate the model on multi-move dataset.
    
    Args:
        model: The FoundationalModel to evaluate
        dataloader: DataLoader for evaluation data
        loss_fn: Loss function
        device: Device to evaluate on (cuda/cpu)
        max_sequence_length: Maximum sequence length in dataset
    
    Returns:
        tuple: (average_loss, accuracy, perfect_rate)
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    perfect_predictions = 0
    num_batches = 0
    
    # Color indices (the last 6 in vocabulary)
    color_indices = list(range(len(VOCAB) - 6, len(VOCAB)))
    
    # Track sequence length distribution in evaluation
    seq_length_counts = {}
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        
        for initial_state, move_sequences, final_state in pbar:
            initial_state = initial_state.to(device)
            move_sequences = move_sequences.to(device)
            final_state = final_state.to(device)
            
            batch_size = initial_state.shape[0]
            
            # Process each sample in the batch (sequences can have different lengths)
            batch_loss = 0
            
            for i in range(batch_size):
                # Get actual sequence length (remove padding if any)
                seq_len = (move_sequences[i] != 0).sum().item()
                if seq_len == 0:  # Skip if no moves
                    continue
                
                # Track sequence length distribution
                seq_length_counts[seq_len] = seq_length_counts.get(seq_len, 0) + 1
                
                # Apply moves iteratively
                current_state = initial_state[i:i+1]
                for move_idx in range(seq_len):
                    move = move_sequences[i, move_idx:move_idx+1]
                    output_logits = model(current_state, move)
                    
                    # Update current state for next iteration
                    color_logits = output_logits[:, :, color_indices]
                    predicted_state = torch.argmax(color_logits, dim=-1) + color_indices[0]
                    current_state = predicted_state
                
                # Extract only color logits from final prediction
                color_logits = output_logits[:, :, color_indices]
                
                # Calculate loss
                final_state_colors = final_state[i:i+1] - color_indices[0]
                sample_loss = loss_fn(color_logits.view(-1, len(color_indices)), final_state_colors.view(-1))
                batch_loss += sample_loss
                
                # Calculate accuracy
                predictions = torch.argmax(color_logits, dim=-1)
                sample_correct = (predictions == final_state_colors).sum().item()
                correct_predictions += sample_correct
                total_predictions += final_state_colors.numel()
                
                # Check for perfect prediction
                if torch.all(predictions == final_state_colors):
                    perfect_predictions += 1
            
            total_loss += batch_loss.item() / batch_size
            num_batches += 1
            
            # Update progress bar
            current_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            current_perfect = perfect_predictions / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0
            pbar.set_postfix({
                'loss': f'{total_loss/num_batches:.4f}',
                'acc': f'{current_accuracy:.2%}',
                'perfect': f'{current_perfect:.1%}'
            })
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    perfect_rate = perfect_predictions / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0
    
    # Print sequence length distribution
    if seq_length_counts:
        total_seqs = sum(seq_length_counts.values())
        print(f"   Sequence length distribution:")
        for length in sorted(seq_length_counts.keys()):
            count = seq_length_counts[length]
            percentage = count / total_seqs * 100
            print(f"     {length} moves: {count:4d} samples ({percentage:4.1f}%)")
    
    return avg_loss, accuracy, perfect_rate

def calculate_intermediate_states(initial_state, move_sequence, seq_len, device):
    """
    Calculate ground truth intermediate states for teacher forcing.
    
    SIMPLIFIED: Since we can't easily set cube state, we'll use a different approach.
    We'll apply the same iterative prediction but use the ground truth final state
    as reference and work backwards or use the dataset's ground truth.
    
    For now, let's use a simplified teacher forcing that just uses the final state
    as the target for the last move, and predicted states for intermediate moves.
    """
    # For now, return None to indicate we should use a simpler teacher forcing approach
    return None

def collate_multi_move(batch):
    """
    Custom collate function for multi-move sequences with variable lengths.
    
    Args:
        batch: List of (initial_state, move_sequence, final_state) tuples
    
    Returns:
        Batched tensors with padded sequences
    """
    initial_states, move_sequences, final_states = zip(*batch)
    
    # Stack initial and final states (fixed size)
    initial_states = torch.stack(initial_states)
    final_states = torch.stack(final_states)
    
    # For now, we'll process sequences individually due to variable lengths
    # In a more advanced implementation, we could pad sequences
    max_len = max(len(seq) for seq in move_sequences)
    
    # Pad sequences to max length (use 0 as padding token)
    padded_sequences = []
    for seq in move_sequences:
        if len(seq) < max_len:
            padding = torch.zeros(max_len - len(seq), dtype=torch.long)
            padded_seq = torch.cat([seq, padding])
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
    
    move_sequences = torch.stack(padded_sequences)
    
    return initial_states, move_sequences, final_states

def train_multi_move_model(model, train_dataset, eval_dataloader, loss_fn, optimizer, scheduler, device, config):
    """
    Train the model on multi-move sequences.
    """
    print(f"Starting multi-move fine-tuning with {len(train_dataset):,} samples...")
    print(f"Configuration: {config}")
    
    # Color indices for proper loss calculation
    color_indices = list(range(len(VOCAB) - 6, len(VOCAB)))
    
    # Training state
    best_accuracy = 0.0
    patience_counter = 0
    train_losses = []
    eval_losses = []
    eval_accuracies = []
    eval_perfect_rates = []
    
    max_seq_length = train_dataset.get_max_sequence_length() if hasattr(train_dataset, 'get_max_sequence_length') else 8
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Update curriculum if using curriculum dataset
        if hasattr(train_dataset, 'update_curriculum'):
            progress = epoch / config['num_epochs']
            curriculum_updated = train_dataset.update_curriculum(progress)
            if curriculum_updated:
                print(f"Curriculum updated - Max sequence length: {train_dataset.get_max_sequence_length()}")
                max_seq_length = train_dataset.get_max_sequence_length()
                
                # IMPORTANT: Align evaluation dataset with new curriculum
                if hasattr(eval_dataloader.dataset, 'align_with_curriculum'):
                    eval_aligned = eval_dataloader.dataset.align_with_curriculum(train_dataset)
                    if eval_aligned:
                        print(f"Evaluation dataset aligned with curriculum")
        
        # Training phase
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        num_batches = 0
        train_seq_counts = {}
        
        # Create dataloader for this epoch
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True,
            collate_fn=collate_multi_move
        )
        
        # Training loop
        pbar = tqdm(train_dataloader, desc=f"Training")
        
        for initial_state, move_sequences, final_state in pbar:
            initial_state = initial_state.to(device)
            move_sequences = move_sequences.to(device)
            final_state = final_state.to(device)
            
            optimizer.zero_grad()
            
            batch_size = initial_state.shape[0]
            batch_loss = 0
            
            # Process each sample in the batch
            for i in range(batch_size):
                # Find actual sequence length (excluding padding)
                seq_len = (move_sequences[i] != 0).sum().item()
                if seq_len == 0:  # Skip if no moves
                    continue
                
                # Track training sequence lengths
                train_seq_counts[seq_len] = train_seq_counts.get(seq_len, 0) + 1
                
                # FAST TRAINING: Focus on final state with better error handling
                current_state = initial_state[i:i+1]
                
                for move_idx in range(seq_len):
                    move = move_sequences[i, move_idx:move_idx+1]
                    output_logits = model(current_state, move)
                    
                    # Update current state for next iteration
                    color_logits = output_logits[:, :, color_indices]
                    predicted_state = torch.argmax(color_logits, dim=-1) + color_indices[0]
                    current_state = predicted_state
                
                # Calculate loss on final prediction only
                color_logits = output_logits[:, :, color_indices]
                final_state_colors = final_state[i:i+1] - color_indices[0]
                
                sample_loss = loss_fn(color_logits.view(-1, len(color_indices)), final_state_colors.view(-1))
                batch_loss += sample_loss
                
                # Calculate accuracy on final prediction
                predictions = torch.argmax(color_logits, dim=-1)
                sample_correct = (predictions == final_state_colors).sum().item()
                epoch_correct += sample_correct
                epoch_total += final_state_colors.numel()
            
            # Backpropagation
            if batch_loss > 0:
                avg_batch_loss = batch_loss / batch_size
                avg_batch_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                
                optimizer.step()
                
                epoch_loss += avg_batch_loss.item()
                num_batches += 1
            
            # Update progress bar
            current_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
            pbar.set_postfix({'loss': f'{epoch_loss/max(1,num_batches):.4f}', 'acc': f'{current_acc:.2%}'})
        
        # Calculate epoch metrics
        epoch_train_loss = epoch_loss / max(1, num_batches)
        epoch_train_acc = epoch_correct / max(1, epoch_total)
        train_losses.append(epoch_train_loss)
        
        # Print training sequence distribution
        if train_seq_counts:
            total_train_seqs = sum(train_seq_counts.values())
            print(f"   Training sequence distribution:")
            for length in sorted(train_seq_counts.keys()):
                count = train_seq_counts[length]
                percentage = count / total_train_seqs * 100
                print(f"     {length} moves: {count:4d} samples ({percentage:4.1f}%)")
        
        # Evaluation phase
        eval_loss, eval_accuracy, eval_perfect_rate = evaluate_multi_model(
            model, eval_dataloader, loss_fn, device, max_seq_length
        )
        eval_losses.append(eval_loss)
        eval_accuracies.append(eval_accuracy)
        eval_perfect_rates.append(eval_perfect_rate)
        
        print(f"Results: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2%}, "
              f"Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.2%}, Eval Perfect: {eval_perfect_rate:.2%}")
        
        # Learning rate scheduling
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(eval_loss)
        else:
            scheduler.step()
        
        # Check for improvement and save best model
        # IMPORTANT: Reset patience when curriculum updates to allow learning new difficulties
        if curriculum_updated:
            patience_counter = 0  # Reset patience for new curriculum level
            print(f"Patience counter reset due to curriculum update")
        elif eval_accuracy > best_accuracy:
            best_accuracy = eval_accuracy
            patience_counter = 0
            print(f"New best accuracy: {best_accuracy:.2%}! Saving model...")
            torch.save(model.state_dict(), "checkpoints/best_multi_move_model.pth")
        else:
            patience_counter += 1
        
        # Early stopping - but only if we've reached max curriculum level
        max_possible_length = getattr(train_dataset, 'final_max_length', 6)
        current_max_length = getattr(train_dataset, 'current_max_length', 2)
        
        if patience_counter >= config['early_stopping_patience']:
            if current_max_length >= max_possible_length:
                print(f"Early stopping after {patience_counter} epochs without improvement (max curriculum reached)")
                break
            else:
                print(f"Patience limit reached but curriculum can still progress (current: {current_max_length}, max: {max_possible_length})")
                patience_counter = max(0, patience_counter - 2)  # Reduce patience but don't reset completely
        
        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/multi_move_checkpoint_epoch_{epoch+1}.pth")
    
    print(f"\nMulti-move fine-tuning completed! Best accuracy: {best_accuracy:.2%}")
    
    # Load best model for final evaluation
    if os.path.exists("checkpoints/best_multi_move_model.pth"):
        model.load_state_dict(torch.load("checkpoints/best_multi_move_model.pth"))
        print(f"Loaded best multi-move model for final evaluation")
    
    return {
        'best_accuracy': best_accuracy,
        'train_losses': train_losses,
        'eval_losses': eval_losses,
        'eval_accuracies': eval_accuracies,
        'eval_perfect_rates': eval_perfect_rates
    }

def main():
    """Main fine-tuning function."""
    # Hyperparameters
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VOCAB_SIZE = len(VOCAB)
    BATCH_SIZE = 16  # Smaller batch size for multi-move sequences
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.0001  # Lower learning rate for fine-tuning
    
    # Model architecture (same as original)
    D_MODEL = 256
    N_HEAD = 8
    D_HID = 1024
    N_ENCODER_LAYERS = 4
    N_DECODER_LAYERS = 2
    DROPOUT = 0.1
    
    # Training parameters
    MAX_GRAD_NORM = 1.0
    EARLY_STOPPING_PATIENCE = 8
    
    # Dataset sizes
    TRAIN_SIZE = 10000  # Smaller dataset for fine-tuning
    EVAL_SIZE = 2000
    
    print(f"Device: {DEVICE}")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Model: d_model={D_MODEL}, nhead={N_HEAD}, d_hid={D_HID}")
    print(f"Datasets: Train={TRAIN_SIZE:,}, Eval={EVAL_SIZE:,}")
    
    # Create datasets
    print(f"\nCreating multi-move datasets...")
    
    # Use curriculum learning for training
    train_dataset = CurriculumMultiMoveDataset(
        TRAIN_SIZE,
        initial_max_length=2,
        final_max_length=6,  # Start conservative
        seed=42
    )
    
    # Aligned evaluation dataset that matches curriculum
    eval_dataset = AlignedEvalMultiMoveDataset(
        EVAL_SIZE,
        train_dataset=train_dataset,
        seed=999
    )
    
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        collate_fn=collate_multi_move
    )
    
    # Load pre-trained model
    print(f"\nLoading pre-trained model...")
    model = PositionAwareFoundationalModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=N_HEAD,
        d_hid=D_HID,
        n_encoder_layers=N_ENCODER_LAYERS,
        n_decoder_layers=N_DECODER_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # Load the best single-move model
    pretrained_path = "checkpoints/best_model.pth"
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))
        print(f"Loaded pre-trained model: {pretrained_path}")
    else:
        print(f"WARNING: Pre-trained model not found at {pretrained_path}")
        print("Starting from scratch...")
    
    print(f"NEW MULTI-MOVE FINE-TUNING FEATURES:")
    print(f"   - Curriculum learning (2→6 move sequences)")
    print(f"   - Lower learning rate for fine-tuning")
    print(f"   - Iterative move application")
    
    # Setup training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    config = {
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'max_grad_norm': MAX_GRAD_NORM,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE
    }
    
    # Fine-tune the model
    print(f"\nStarting multi-move fine-tuning...")
    results = train_multi_move_model(
        model, train_dataset, eval_dataloader, loss_fn, optimizer, scheduler, DEVICE, config
    )
    
    print(f"\nFine-tuning complete!")
    print(f"Final Results:")
    print(f"   - Best Multi-Move Accuracy: {results['best_accuracy']:.2%}")
    print(f"   - Model saved as: checkpoints/best_multi_move_model.pth")

if __name__ == "__main__":
    main()
