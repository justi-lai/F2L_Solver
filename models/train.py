import os
import sys
import glob
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.general_model import CubeSolver
from models.dataset import CubeDataset

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data (File Split)
    if not os.path.exists(args.data_path):
        raise ValueError(f"Data path {args.data_path} not found.")
        
    all_files = sorted(glob.glob(os.path.join(args.data_path, "part_*.pt")))
    if not all_files:
        raise ValueError("No part files found.")
        
    random.seed(42) # Deterministic Split
    random.shuffle(all_files)
    
    split_idx = int(0.9 * len(all_files))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"Total Shards: {len(all_files)}. Train: {len(train_files)}, Val: {len(val_files)}")
    
    train_dataset = CubeDataset(train_files)
    val_dataset = CubeDataset(val_files)
    
    # 2. DataLoaders (Shuffle=False for Iterable)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    
    # 3. Init Model
    model = CubeSolver(vocab_size=38).to(device)
    
    # 4. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    best_val_loss = float('inf')
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 5. Training Loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Pbar (estimate length)
        # Note: IterableDataset length is just estimate. 
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_states, batch_actions, batch_mask in pbar:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            
            inputs = batch_states.view(batch_states.size(0), batch_states.size(1), -1)
            
            optimizer.zero_grad()
            logits = model(inputs)
            
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = batch_actions.view(-1)
            
            loss = criterion(logits_flat, targets_flat)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            
            train_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            mask = targets_flat != 0
            correct = (preds.view(-1) == targets_flat) & mask
            train_correct += correct.sum().item()
            train_total += mask.sum().item()
            
            batch_acc = correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0
            pbar.set_postfix({'loss': loss.item(), 'acc': batch_acc})
            
        avg_train_loss = train_loss / len(pbar) 
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch_states, batch_actions, batch_mask in val_loader:
                batch_states = batch_states.to(device)
                batch_actions = batch_actions.to(device)
                
                inputs = batch_states.view(batch_states.size(0), batch_states.size(1), -1)
                logits = model(inputs)
                
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = batch_actions.view(-1)
                
                loss = criterion(logits_flat, targets_flat)
                val_loss += loss.item()
                val_steps += 1
                
                preds = torch.argmax(logits, dim=-1)
                mask = targets_flat != 0
                correct = (preds.view(-1) == targets_flat) & mask
                val_correct += correct.sum().item()
                val_total += mask.sum().item()
        
        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))
            print(f"  Saved best model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4) # AdamW sensitive
    parser.add_argument("--weight_decay", type=float, default=1e-2) # L2
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--data_path", type=str, default="data/dataset_1m")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    
    args = parser.parse_args()
    train(args)
