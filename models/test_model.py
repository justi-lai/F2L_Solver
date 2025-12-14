import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import sys
from tqdm import tqdm
import random
import copy
import heapq

# Add path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PyCube-Solver"))

from models.general_model import CubeSolver
from models.dataset import CubeDataset
from data_generation.cube_generator import CubeGenerator
from data_generation.virtual_cube import VirtualCube
# Adjusted imports: Since PyCube-Solver is in sys.path now, we import directly from files
from cube import Cube
from solver import Solver
from helper import getScramble

class BeamNode:
    def __init__(self, cube, history_string, states_history, score, finished=False):
        self.cube = cube # Cube object
        self.history_string = history_string # List of move strings
        self.states_history = states_history # List of tensors
        self.score = score # Cumulative log prob
        self.finished = finished

    def __lt__(self, other):
        # Max heap simulation (invert score for min heap or just verify sort order)
        # We want higher scores first.
        return self.score > other.score

def check_solved(cube):
    faces = cube.getFaces()
    for face in faces:
        c = face[1][1] # Center
        for row in face:
            for col in row:
                if col != c:
                    return False
    return True

def beam_search_solve(model, device, beam_width=5, max_steps=150):
    print(f"\n--- Beam Search (Width={beam_width}) ---")
    
    gen = CubeGenerator()
    scramble_str = getScramble(20)
    
    # Format scramble for readability (space separated)
    # The format from getScramble is compact (e.g. "R'U2F")
    # We want "R' U2 F"
    spaced_scramble = ""
    i = 0
    while i < len(scramble_str):
        spaced_scramble += scramble_str[i]
        i += 1
        # Check for modifier ('2' or "'")
        if i < len(scramble_str) and (scramble_str[i] == "2" or scramble_str[i] == "'"):
            spaced_scramble += scramble_str[i]
            i += 1
        spaced_scramble += " "
    
    print(f"Scramble: {spaced_scramble.strip()}")
    
    # Init root
    root_cube = Cube()
    root_cube.doMoves(scramble_str)
    print("Initial Cube State:")
    print(root_cube)
    
    vc = VirtualCube(root_cube)
    root_node = BeamNode(
        cube=root_cube,
        history_string=[],
        states_history=[vc.get_one_hot_tensor()],
        score=0.0
    )
    
    beam = [root_node]
    
    for step in range(max_steps):
        candidates = []
        
        # Expand each node
        for node in beam:
            if node.finished:
                candidates.append(node) # Keep finished nodes? Usually we remove them to "done" list.
                continue
                
            # Prepare Input
            input_stack = torch.stack(node.states_history).unsqueeze(0).to(device)
            B, S, _, _ = input_stack.shape
            input_seq = input_stack.view(B, S, -1)
            
            # Forward
            with torch.no_grad():
                # Get logits for last step: [1, Seq, Vocab] -> [1, Vocab]
                logits = model(input_seq)[:, -1, :] 
                log_probs = torch.log_softmax(logits, dim=-1)
                
            # Top K
            top_scores, top_indices = torch.topk(log_probs, k=beam_width, dim=-1)
            
            # Create branches
            for i in range(beam_width):
                token_id = top_indices[0, i].item()
                token_score = top_scores[0, i].item()
                
                # Check tokens
                if token_id == 0: # PAD
                    continue
                if token_id == 1: # EOS -> Finished
                    finished_node = BeamNode(
                        cube=node.cube, # State doesn't change on EOS
                        history_string=node.history_string,
                        states_history=node.states_history,
                        score=node.score + token_score,
                        finished=True
                    )
                    candidates.append(finished_node)
                    continue
                    
                move_str = gen.id_to_token.get(token_id)
                if not move_str: continue
                
                # New Cube State
                # MUST DEEP COPY CUBE
                new_cube = copy.deepcopy(node.cube)
                pycube_move = move_str.replace("P", "'")
                new_cube.doMoves(pycube_move)
                
                # Check Solved ONLY if we want early exit (or assign massive bonus?)
                # We can perform check_solved here. If solved, we found it!
                if check_solved(new_cube):
                    print(f"SOLVED in {step+1} steps!")
                    final_solution = node.history_string + [move_str]
                    print(f"Solution: {' '.join(final_solution)}")
                    print("Final Cube State:")
                    print(new_cube)
                    return
                
                # New VC
                new_vc = VirtualCube(new_cube)
                
                # New Node
                new_node = BeamNode(
                    cube=new_cube,
                    history_string=node.history_string + [move_str],
                    states_history=node.states_history + [new_vc.get_one_hot_tensor()],
                    score=node.score + token_score
                )
                candidates.append(new_node)
        
        # Prune
        # Sort by score desc
        candidates.sort(key=lambda x: x.score, reverse=True)
        beam = candidates[:beam_width]
        
        # Logging best so far
        if step % 5 == 0:
            print(f"Step {step}: Best Score={beam[0].score:.4f} Moves={len(beam[0].history_string)}")
            
    print("Beam search failed to find solution.")
    print(f"Best Attempt ({beam[0].score:.4f}): {' '.join(beam[0].history_string)}")

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    model = CubeSolver(vocab_size=38).to(device)
    
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Error: Checkpoint {args.checkpoint} not found!")
        return

    model.eval()

    if args.interactive:
        beam_search_solve(model, device, beam_width=args.beam_width)
        return

    # Dataset Evaluation
    dataset = CubeDataset(args.data_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    total_correct = 0
    total_tokens = 0
    
    print("Starting evaluation...")
    
    pbar = tqdm(loader, desc="Testing")
    
    with torch.no_grad():
        for batch_states, batch_actions, batch_mask in pbar:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            
            inputs = batch_states.view(batch_states.size(0), batch_states.size(1), -1)
            logits = model(inputs)
            
            preds = torch.argmax(logits, dim=-1)
            
            mask = batch_actions != 0
            
            correct = (preds == batch_actions) & mask
            
            batch_cor = correct.sum().item()
            batch_tot = mask.sum().item()
            
            total_correct += batch_cor
            total_tokens += batch_tot
            
            acc = batch_cor / batch_tot if batch_tot > 0 else 0
            pbar.set_postfix({'acc': acc})
            
    acc = total_correct / total_tokens if total_tokens > 0 else 0
    print(f"\nTest Results:")
    print(f"Total Tokens Evaluated: {total_tokens}")
    print(f"Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_path", type=str, default="data/dataset.pt")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--interactive", action="store_true", help="Run a live solve on a random scramble")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width for search (default: 5)")
    
    args = parser.parse_args()
    test(args)
