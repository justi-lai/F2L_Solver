"""
Interactive Multi-Move Model Testing Script

This script allows you to:
1. Test the model on random scrambled cubes with move sequences
2. Test on a solved cube to manually verify results
3. See visual representations of cube states
4. Apply custom move sequences
5. Compare predicted vs actual results
"""

import torch
import torch.nn as nn
import os
import sys
import numpy as np
from cuber.cuber import Cube

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import VOCAB, COLOR_MAP
from models.general_model import PositionAwareFoundationalModel
from data_generation.cube_state_gen import generate_random_scramble

class InteractiveMultiMoveTester:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.load_model()
        
        # Get move tokens
        all_tokens = list(VOCAB.keys())
        self.move_tokens = [token for token in all_tokens if token not in ['WHITE', 'YELLOW', 'GREEN', 'BLUE', 'RED', 'ORANGE']]
        self.color_indices = list(range(len(VOCAB) - 6, len(VOCAB)))
        
        print("🎯 Interactive Multi-Move Model Tester")
        print("="*50)
        print(f"Device: {self.device}")
        print(f"Available moves: {len(self.move_tokens)} total")
        print("✅ Ready to test!")
    
    def load_model(self):
        """Load the best multi-move model"""
        self.model = PositionAwareFoundationalModel(
            vocab_size=len(VOCAB),
            d_model=256,
            nhead=8,
            d_hid=1024,
            n_encoder_layers=4,
            n_decoder_layers=2,
            dropout=0.1
        ).to(self.device)
        
        model_path = "checkpoints/best_multi_move_model.pth"  # Use the best performing checkpoint
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Model loaded from {model_path}")
        else:
            print(f"❌ ERROR: {model_path} not found!")
            sys.exit(1)
        
        self.model.eval()
    
    def cube_state_to_visual(self, state_str):
        """Convert cube state string to a visual representation"""
        faces = state_str.split(" ")
        
        # Define color mappings for better visualization
        color_map = {
            'W': '⬜', 'Y': '🟨', 'G': '🟩', 
            'B': '🟦', 'R': '🟥', 'O': '🟧'
        }
        
        visual = []
        face_names = ['U (Up)', 'L (Left)', 'F (Front)', 'R (Right)', 'B (Back)', 'D (Down)']
        
        for i, face_name in enumerate(face_names):
            face_start = i * 9
            face_colors = faces[face_start:face_start + 9]
            
            visual.append(f"  {face_name}:")
            # Display 3x3 grid
            for row in range(3):
                row_start = row * 3
                row_colors = face_colors[row_start:row_start + 3]
                visual_row = " ".join([color_map.get(color, color) for color in row_colors])
                visual.append(f"    {visual_row}")
            visual.append("")
        
        return "\n".join(visual)
    
    def state_str_to_tensor(self, state_str):
        """Convert state string to tensor"""
        return torch.tensor([VOCAB[COLOR_MAP[face]] for face in state_str.split(" ")], 
                          dtype=torch.long).unsqueeze(0).to(self.device)
    
    def moves_to_tensor(self, moves):
        """Convert move list to tensor"""
        return torch.tensor([VOCAB[move] for move in moves], 
                          dtype=torch.long).unsqueeze(0).to(self.device)
    
    def predict_sequence(self, initial_state_str, moves):
        """Predict the result of applying moves to initial state"""
        initial_tensor = self.state_str_to_tensor(initial_state_str)
        move_tensors = self.moves_to_tensor(moves)
        
        with torch.no_grad():
            current_state = initial_tensor
            
            # Apply moves iteratively
            for move_idx in range(len(moves)):
                move = move_tensors[:, move_idx:move_idx+1]
                output_logits = self.model(current_state, move.squeeze(-1))
                
                # Get predicted next state
                color_logits = output_logits[:, :, self.color_indices]
                predicted_state = torch.argmax(color_logits, dim=-1) + self.color_indices[0]
                current_state = predicted_state
            
            # Convert back to state string
            predicted_indices = current_state.cpu().numpy()[0]
            reverse_vocab = {v: k for k, v in VOCAB.items()}
            reverse_color_map = {v: k for k, v in COLOR_MAP.items()}
            
            predicted_faces = []
            for idx in predicted_indices:
                color_token = reverse_vocab[idx]
                color_str = reverse_color_map[color_token]
                predicted_faces.append(color_str)
            
            return " ".join(predicted_faces)
    
    def get_actual_result(self, initial_state_str, moves):
        """Get the actual result by applying moves to a real cube"""
        cube = Cube()
        
        # Set the cube to the initial state
        # Since we can't directly set state, we'll use the cube's current state
        # This is a limitation - for solved cube we can start fresh
        if initial_state_str == self.get_solved_state():
            # For solved cube, start fresh
            pass
        else:
            # For scrambled states, we assume the cube is in the right state
            # This is a simplification for the demo
            pass
        
        # Apply the moves
        for move in moves:
            cube.turn(move)
        
        return cube.get_faces_str("ULFRBD")
    
    def get_solved_state(self):
        """Get the solved cube state"""
        cube = Cube()
        return cube.get_faces_str("ULFRBD")
    
    def test_random_scramble(self, num_moves=None):
        """Test on a random scrambled cube"""
        print("\n🎲 Testing on Random Scrambled Cube")
        print("-" * 40)
        
        # Generate random scramble
        scramble_length = np.random.randint(5, 15)
        scramble = generate_random_scramble(scramble_length)
        
        # Create initial state
        cube = Cube()
        cube.turn(scramble)
        initial_state = cube.get_faces_str("ULFRBD")
        
        # Generate random move sequence
        if num_moves is None:
            num_moves = np.random.randint(1, 6)  # 1-5 moves
        
        moves = [np.random.choice(self.move_tokens) for _ in range(num_moves)]
        
        print(f"📝 Initial scramble: {scramble}")
        print(f"🎯 Test moves: {' '.join(moves)} ({len(moves)} moves)")
        print()
        
        # Show initial state
        print("🔵 INITIAL STATE:")
        print(self.cube_state_to_visual(initial_state))
        
        # Get actual result
        cube.turn(' '.join(moves))
        actual_result = cube.get_faces_str("ULFRBD")
        
        # Get predicted result
        predicted_result = self.predict_sequence(initial_state, moves)
        
        # Show results
        print("🤖 PREDICTED RESULT:")
        print(self.cube_state_to_visual(predicted_result))
        
        print("✅ ACTUAL RESULT:")
        print(self.cube_state_to_visual(actual_result))
        
        # Check if they match
        if predicted_result == actual_result:
            print("🎉 PERFECT MATCH! ✅")
        else:
            print("❌ MISMATCH!")
            
            # Count differences
            pred_faces = predicted_result.split(" ")
            actual_faces = actual_result.split(" ")
            differences = sum(1 for p, a in zip(pred_faces, actual_faces) if p != a)
            print(f"📊 Differences: {differences}/54 stickers ({differences/54*100:.1f}%)")
        
        return predicted_result == actual_result
    
    def test_solved_cube(self, moves=None):
        """Test on a solved cube with custom moves"""
        print("\n🟦 Testing on Solved Cube")
        print("-" * 40)
        
        initial_state = self.get_solved_state()
        
        if moves is None:
            # Ask user for moves
            print("Available moves:", ", ".join(self.move_tokens[:20]), "...")
            move_input = input("Enter moves (space-separated, or press Enter for random): ").strip()
            
            if move_input:
                moves = move_input.split()
                # Validate moves
                invalid_moves = [m for m in moves if m not in self.move_tokens]
                if invalid_moves:
                    print(f"❌ Invalid moves: {invalid_moves}")
                    return False
            else:
                # Random moves
                num_moves = np.random.randint(1, 4)  # 1-3 moves for easier verification
                moves = [np.random.choice(self.move_tokens) for _ in range(num_moves)]
        
        print(f"🎯 Applying moves: {' '.join(moves)} ({len(moves)} moves)")
        print()
        
        # Show initial state (solved)
        print("🔵 INITIAL STATE (SOLVED):")
        print(self.cube_state_to_visual(initial_state))
        
        # Get actual result
        cube = Cube()
        cube.turn(' '.join(moves))
        actual_result = cube.get_faces_str("ULFRBD")
        
        # Get predicted result
        predicted_result = self.predict_sequence(initial_state, moves)
        
        # Show results
        print("🤖 PREDICTED RESULT:")
        print(self.cube_state_to_visual(predicted_result))
        
        print("✅ ACTUAL RESULT:")
        print(self.cube_state_to_visual(actual_result))
        
        # Check if they match
        if predicted_result == actual_result:
            print("🎉 PERFECT MATCH! ✅")
        else:
            print("❌ MISMATCH!")
            
            # Count differences
            pred_faces = predicted_result.split(" ")
            actual_faces = actual_result.split(" ")
            differences = sum(1 for p, a in zip(pred_faces, actual_faces) if p != a)
            print(f"📊 Differences: {differences}/54 stickers ({differences/54*100:.1f}%)")
        
        return predicted_result == actual_result
    
    def run_interactive_session(self):
        """Run the interactive testing session"""
        print("\n" + "="*50)
        print("🎮 INTERACTIVE TESTING SESSION")
        print("="*50)
        
        while True:
            print("\nChoose an option:")
            print("1. 🎲 Test random scrambled cube")
            print("2. 🟦 Test solved cube (enter your own moves)")
            print("3. 🟦 Test solved cube (random moves)")
            print("4. 🎯 Run multiple random tests")
            print("5. ❌ Exit")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                self.test_random_scramble()
                
            elif choice == '2':
                self.test_solved_cube()
                
            elif choice == '3':
                moves = [np.random.choice(self.move_tokens) for _ in range(np.random.randint(1, 4))]
                self.test_solved_cube(moves)
                
            elif choice == '4':
                num_tests = int(input("How many tests? (default 5): ") or "5")
                print(f"\n🔄 Running {num_tests} random tests...")
                
                successes = 0
                for i in range(num_tests):
                    print(f"\n--- Test {i+1}/{num_tests} ---")
                    if self.test_random_scramble():
                        successes += 1
                
                print(f"\n📊 SUMMARY: {successes}/{num_tests} perfect matches ({successes/num_tests*100:.1f}%)")
                
            elif choice == '5':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-5.")

def main():
    """Main function"""
    try:
        tester = InteractiveMultiMoveTester()
        tester.run_interactive_session()
        
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
