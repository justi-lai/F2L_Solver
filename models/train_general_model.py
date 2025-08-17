import torch
import torch.nn as nn
import os
import sys
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import PreTrainDataset, AdvancedCurriculumDataset, HybridDataset, VOCAB, COLOR_MAP
from models.general_model import PositionAwareFoundationalModel, FoundationalModel

def evaluate_model(model, dataloader, loss_fn, device):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The FoundationalModel to evaluate
        dataloader: DataLoader for evaluation data
        loss_fn: Loss function
        device: Device to evaluate on (cuda/cpu)
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    num_batches = 0
    
    # Color indices (the last 6 in vocabulary: WHITE, YELLOW, GREEN, BLUE, RED, ORANGE)
    color_indices = list(range(len(VOCAB) - 6, len(VOCAB)))
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        
        for initial_state, move, final_state in pbar:
            initial_state = initial_state.to(device)
            move = move.to(device)
            final_state = final_state.to(device)
            
            # Forward pass
            output_logits = model(initial_state, move)
            
            # Extract only color logits (last 6 dimensions)
            color_logits = output_logits[:, :, color_indices]  # Shape: (batch, 54, 6)
            
            # Calculate loss only on color predictions
            # Adjust targets to be in range [0, 5] instead of [54, 59]
            color_targets = final_state - color_indices[0]  # Subtract 54 to get [0, 5]
            loss = loss_fn(color_logits.view(-1, 6), color_targets.view(-1))
            total_loss += loss.item()
            num_batches += 1
            
            # Calculate accuracy on color predictions
            color_predictions = torch.argmax(color_logits, dim=-1) + color_indices[0]  # Convert back to [54, 59]
            correct_predictions += (color_predictions == final_state).sum().item()
            total_predictions += final_state.numel()
            
            # Update progress bar
            current_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            pbar.set_postfix({
                'loss': f'{total_loss/num_batches:.4f}',
                'acc': f'{current_accuracy:.2%}'
            })
    
    average_loss = total_loss / num_batches if num_batches > 0 else 0
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    return average_loss, accuracy

def test_model(model, test_dataloader, device):
    """
    Test the model and show some example predictions.
    
    Args:
        model: The FoundationalModel to test
        test_dataloader: DataLoader for test data
        device: Device to test on (cuda/cpu)
    """
    model.eval()
    
    print(f"Testing model on {len(test_dataloader.dataset)} samples...")
    
    # Test overall performance
    loss_fn = nn.CrossEntropyLoss()
    test_loss, test_accuracy = evaluate_model(model, test_dataloader, loss_fn, device)
    
    print(f"Test Results:")
    print(f"   - Loss: {test_loss:.4f}")
    print(f"   - Accuracy: {test_accuracy:.2%}")
    
    # Show some example predictions
    show_examples(model, test_dataloader, device, num_examples=3)

def show_examples(model, dataloader, device, num_examples=3):
    """
    Show some example predictions from the model.
    
    Args:
        model: The FoundationalModel
        dataloader: DataLoader to sample from
        device: Device for inference
        num_examples: Number of examples to show
    """
    model.eval()
    
    print(f"\n🔍 Example Predictions:")
    
    # Get one batch
    for batch_idx, (initial_state, move, final_state) in enumerate(dataloader):
        if batch_idx > 0:  # Only use first batch
            break
            
        initial_state = initial_state.to(device)
        move = move.to(device)
        final_state = final_state.to(device)
        
        # Color indices for proper prediction
        color_indices = list(range(len(VOCAB) - 6, len(VOCAB)))
        
        with torch.no_grad():
            output_logits = model(initial_state, move)
            color_logits = output_logits[:, :, color_indices]
            predictions = torch.argmax(color_logits, dim=-1) + color_indices[0]
        
        # Show first few examples
        for i in range(min(num_examples, initial_state.size(0))):
            print(f"\n   Example {i+1}:")
            
            # Convert move index back to move name
            move_idx = move[i].item()
            move_name = "UNKNOWN"
            for token, idx in VOCAB.items():
                if idx == move_idx:
                    move_name = token
                    break
            
            print(f"   Move: {move_name}")
            
            # Show accuracy for this sample
            sample_correct = (predictions[i] == final_state[i]).sum().item()
            sample_accuracy = sample_correct / 54
            print(f"   Accuracy: {sample_accuracy:.2%} ({sample_correct}/54)")
            
            # Show a few specific predictions vs actual
            print(f"   Sample positions (pred vs actual):")
            for pos in [0, 13, 26, 39, 52]:  # A few spread out positions
                pred_color_idx = predictions[i, pos].item()
                actual_color_idx = final_state[i, pos].item()
                
                # Convert back to color names
                pred_color = "UNK"
                actual_color = "UNK"
                for token, idx in VOCAB.items():
                    if idx == pred_color_idx:
                        pred_color = token
                    if idx == actual_color_idx:
                        actual_color = token
                
                match_symbol = "[CORRECT]" if pred_color_idx == actual_color_idx else "[WRONG]"
                print(f"     Pos {pos:2d}: {pred_color:6s} vs {actual_color:6s} {match_symbol}")

def train_model(model, train_dataset, eval_dataloader, loss_fn, optimizer, scheduler, device, config):
    """
    Train the model with curriculum learning and all improvements.
    """
    print(f"Starting training with {len(train_dataset):,} samples...")
    print(f"Configuration: {config}")
    
    # Color indices for proper loss calculation
    color_indices = list(range(len(VOCAB) - 6, len(VOCAB)))
    
    # Training state
    best_accuracy = 0.0
    patience_counter = 0
    train_losses = []
    eval_losses = []
    eval_accuracies = []
    
    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(config['num_epochs']):
        print(f"\n🔄 Epoch {epoch + 1}/{config['num_epochs']}")
        
        # Update curriculum if using curriculum learning
        if hasattr(train_dataset, 'update_curriculum'):
            train_dataset.update_curriculum(epoch, config['num_epochs'])
            if hasattr(train_dataset, 'max_scramble'):
                current_difficulty = train_dataset.max_scramble
                print(f"Current curriculum difficulty: {current_difficulty}")
        elif hasattr(train_dataset, 'curriculum_data'):
            # For HybridDataset
            train_dataset.update_curriculum(epoch, config['num_epochs'])
            if hasattr(train_dataset.curriculum_data, 'max_scramble'):
                current_difficulty = train_dataset.curriculum_data.max_scramble
                print(f"Current curriculum difficulty: {current_difficulty}")
        
        # Training phase
        model.train()
        total_loss = 0
        total_correct = 0
        total_predictions = 0
        num_batches = 0
        
        # Create train dataloader for this epoch
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True
        )
        
        pbar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
        for initial_state, move, final_state in pbar:
            initial_state = initial_state.to(device)
            move = move.to(device)
            final_state = final_state.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output_logits = model(initial_state, move)
            
            # Extract only color logits (last 6 dimensions)
            color_logits = output_logits[:, :, color_indices]  # Shape: (batch, 54, 6)
            
            # Calculate loss only on color predictions
            # Adjust targets to be in range [0, 5] instead of [54, 59]
            color_targets = final_state - color_indices[0]  # Subtract 54 to get [0, 5]
            loss = loss_fn(color_logits.view(-1, 6), color_targets.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if specified
            if config.get('max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Calculate training accuracy
            color_predictions = torch.argmax(color_logits, dim=-1) + color_indices[0]  # Convert back to [54, 59]
            total_correct += (color_predictions == final_state).sum().item()
            total_predictions += final_state.numel()
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            current_acc = total_correct / total_predictions if total_predictions > 0 else 0
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{current_acc:.2%}'})
        
        # Record training loss
        epoch_train_loss = total_loss / num_batches if num_batches > 0 else 0
        epoch_train_acc = total_correct / total_predictions if total_predictions > 0 else 0
        train_losses.append(epoch_train_loss)
        
        # Evaluation phase
        eval_loss, eval_accuracy = evaluate_model(model, eval_dataloader, loss_fn, device)
        eval_losses.append(eval_loss)
        eval_accuracies.append(eval_accuracy)
        
        print(f"Results: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2%}, "
              f"Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.2%}")
        
        # Learning rate scheduling
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(eval_loss)
        else:
            scheduler.step()
        
        # Check for improvement and save best model
        if eval_accuracy > best_accuracy:
            best_accuracy = eval_accuracy
            patience_counter = 0
            print(f"New best accuracy: {best_accuracy:.2%}! Saving model...")
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
        else:
            patience_counter += 1
        
        # Early stopping
        if config.get('early_stopping_patience') and patience_counter >= config['early_stopping_patience']:
            print(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break
        
        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/checkpoint_epoch_{epoch+1}.pth")
    
    print(f"\nTraining completed! Best accuracy: {best_accuracy:.2%}")
    
    # Load best model for final evaluation
    if os.path.exists("checkpoints/best_model.pth"):
        model.load_state_dict(torch.load("checkpoints/best_model.pth"))
        print(f"Loaded best model for final evaluation")
    
    return {
        'best_accuracy': best_accuracy,
        'train_losses': train_losses,
        'eval_losses': eval_losses,
        'eval_accuracies': eval_accuracies
    }

def main():
    print("🤖 Foundational Model Training")
    print("=" * 50)
    
    # Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model hyperparameters
    VOCAB_SIZE = len(VOCAB)
    D_MODEL = 256
    N_HEAD = 8
    D_HID = 1024
    N_ENCODER_LAYERS = 4
    N_SSM_LAYERS = 2
    DROPOUT = 0.1
    
    # Training hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.0002
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0
    EARLY_STOPPING_PATIENCE = 15
    
    # Dataset sizes
    TRAIN_SIZE = 15000
    EVAL_SIZE = 3000
    TEST_SIZE = 1000
    
    print(f"Device: {DEVICE}")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Model: d_model={D_MODEL}, nhead={N_HEAD}, d_hid={D_HID}")
    print(f"Datasets: Train={TRAIN_SIZE:,}, Eval={EVAL_SIZE:,}, Test={TEST_SIZE:,}")
    
    # Create datasets with the FIXED data generation
    print(f"\nCreating fixed datasets...")
    
    # Use curriculum learning for training
    train_dataset = AdvancedCurriculumDataset(
        TRAIN_SIZE,
        min_scramble=1,
        max_scramble=15,
        balanced_moves=True,
        difficulty_progression="exponential",
        seed=42
    )
    
    eval_dataset = PreTrainDataset(EVAL_SIZE, seed=123)
    test_dataset = PreTrainDataset(TEST_SIZE, seed=456)
    
    # Create data loaders
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model - using new position-aware architecture
    print(f"\nCreating position-aware model...")
    model = PositionAwareFoundationalModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=N_HEAD,
        d_hid=D_HID,
        n_encoder_layers=N_ENCODER_LAYERS,
        n_decoder_layers=N_SSM_LAYERS,  # renamed from n_ssm_layers
        dropout=DROPOUT
    ).to(DEVICE)
    
    print(f"NEW FEATURES:")
    print(f"   - Cube-aware positional encoding (faces, positions, sticker types)")
    print(f"   - Encoder-decoder architecture for knowledge transfer")
    print(f"   - Enhanced spatial reasoning capabilities")
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Setup training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    
    # Training configuration
    config = {
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'max_grad_norm': MAX_GRAD_NORM,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE
    }
    
    # Train the model
    print(f"\nStarting training...")
    results = train_model(
        model, train_dataset, eval_dataloader, loss_fn, optimizer, scheduler, DEVICE, config
    )
    
    # Final testing
    print(f"\nFinal testing on unseen data...")
    test_model(model, test_dataloader, DEVICE)
    
    print(f"\nTraining complete!")
    print(f"Final Results:")
    print(f"   - Best Accuracy: {results['best_accuracy']:.2%}")
    print(f"   - Training completed successfully!")

if __name__ == "__main__":
    main()
