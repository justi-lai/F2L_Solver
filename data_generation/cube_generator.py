import os
import sys
import random
import torch

# Add PyCube-Solver to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PyCube-Solver"))
from cube import Cube
from solver import Solver
from helper import getScramble, parseFormula
from data_generation.virtual_cube import VirtualCube

class CubeGenerator:
    """
    Generates Rubik's Cube solves (State, Action) pairs for training.
    """
    def __init__(self):
        # Define Vocabulary
        # 0: PAD, 1: EOS
        # Basic Moves: U, U', D, D', R, R', L, L', F, F', B, B'
        # Middle Moves: M, M', E, E', S, S'
        # Rotations: x, x', y, y', z, z'
        # Wide Moves: u etc.
        
        self.moves_list = [
            'U', 'UP', 'D', 'DP', 'R', 'RP', 'L', 'LP', 'F', 'FP', 'B', 'BP', # Face
            'M', 'MP', 'E', 'EP', 'S', 'SP', # Slice
            'x', 'xP', 'y', 'yP', 'z', 'zP', # Rotation
            'u', 'uP', 'd', 'dP', 'r', 'rP', 'l', 'lP', 'f', 'fP', 'b', 'bP' # Wide
        ]
        
        self.token_to_id = {
            "<PAD>": 0,
            "<EOS>": 1
        }
        self.id_to_token = {
            0: "<PAD>",
            1: "<EOS>"
        }
        
        idx = 2
        for m in self.moves_list:
            self.token_to_id[m] = idx
            self.id_to_token[idx] = m
            idx += 1
            
        self.vocab_size = idx
        
    def get_token_id(self, move_str):
        return self.token_to_id.get(move_str, None)

    def _process_moves_string(self, moves_str):
        """
        Parses a move string into a list of atomic moves (no '2's, handle wide).
        Uses PyCube's parseFormula, then post-processes.
        """
        # Clean string: Solver output contains newlines, and helper doesn't like spaces/newlines.
        moves_str = moves_str.replace('\n', '').replace(' ', '')
        
        raw_list = parseFormula(moves_str, condense=False) 
        
        expanded_list = []
        wide_map = {
            'u': ['U', 'EP'], 'uP': ['UP', 'E'],
            'd': ['D', 'E'], 'dP': ['DP', 'EP'],
            'r': ['R', 'MP'], 'rP': ['RP', 'M'],
            'l': ['L', 'M'], 'lP': ['LP', 'MP'],
            'f': ['F', 'S'], 'fP': ['FP', 'SP'],
            'b': ['B', 'SP'], 'bP': ['BP', 'S']
        }
        
        for m in raw_list:
            if m in wide_map:
                expanded_list.extend(wide_map[m])
            else:
                expanded_list.append(m)
                
        return expanded_list

    def generate_solve(self, scramble_length=20, optimize=False):
        """
        Generates a solve trajectory.
        Returns:
            states (List[Tensor]): Sequence of cube states.
            actions (List[int]): Sequence of action IDs (next move to take).
        """
        # 1. Init Cube & VirtualCube
        c = Cube()
        vc = VirtualCube(c)
        
        # 2. Scramble
        scramble_str = getScramble(scramble_length)
        c.doMoves(scramble_str)
        
        # 3. Solve
        solver = Solver(c)
        solver.solveCube(optimize=optimize)
        solution_str = solver.getMoves(decorated=False)
        
        # 4. Parse Solution
        move_sequence = self._process_moves_string(solution_str)
        
        states = []
        actions = []
        
        # 5. Build Trajectory
        for move_token in move_sequence:
            # Capture state
            state_tensor = vc.get_one_hot_tensor()
            states.append(state_tensor)
            
            # Capture action
            tid = self.get_token_id(move_token)
            if tid is None:
                print(f"Warning: Unknown token {move_token}")
                continue
            actions.append(tid)
            
            # Advance state
            c.doMoves(move_token)
            
        # Add EOS
        # For the final state (solved), the action is EOS.
        states.append(vc.get_one_hot_tensor()) # Solved state
        actions.append(self.token_to_id["<EOS>"])
        
        return states, actions

    def generate_dagger_solve(self, scramble_length=20, epsilon=0.1, max_steps=100, optimize=False):
        """
        Generates a solve with DAgger (perturbations).
        """
        c = Cube()
        vc = VirtualCube(c)
        
        scramble_str = getScramble(scramble_length)
        c.doMoves(scramble_str)
        
        states = []
        actions = []
        
        for _ in range(max_steps):
            # Check if solved
            solver = Solver(c)
            # PyCube ISOLVED check
            if solver.isSolved():
                break
                
            solver.solveCube(optimize=optimize)
            solution_str = solver.getMoves(decorated=False)
            
            if not solution_str:
                break
                
            move_sequence = self._process_moves_string(solution_str)
            if not move_sequence:
                break
                
            optimal_move = move_sequence[0] # Next optimal move
            
            # Capture State
            states.append(vc.get_one_hot_tensor())
            
            # Decide: Optimal or Random
            if random.random() < epsilon:
                # Picker random move (noise)
                # But we RECORD the OPTIMAL move as the target (Teacher Forcing)
                actions.append(self.token_to_id[optimal_move])
                
                # Execute RANDOM move
                random_move = random.choice(self.moves_list)
                c.doMoves(random_move)
                
            else:
                # Optimal
                actions.append(self.token_to_id[optimal_move])
                c.doMoves(optimal_move)
        
        # Add EOS
        states.append(vc.get_one_hot_tensor())
        actions.append(self.token_to_id["<EOS>"])
        
        return states, actions

if __name__ == "__main__":
    # Test
    gen = CubeGenerator()
    print(f"Vocab Size: {gen.vocab_size}")
    
    print("\n--- Generating Standard Solve ---")
    states, actions = gen.generate_solve(5) 
    print(f"Trajectory Length: {len(actions)}")
    print(f"First 5 Actions: {[gen.id_to_token[a] for a in actions[:5]]}")
    
    print("\n--- Generating DAgger Solve ---")
    states_d, actions_d = gen.generate_dagger_solve(5, epsilon=0.5)
    print(f"Trajectory Length: {len(actions_d)}")
    print(f"First 5 Actions: {[gen.id_to_token[a] for a in actions_d[:5]]}")
