import os
import sys
from data_generation.cube_state_gen import (
    generate_random_single_move_data, 
    generate_curriculum_single_move_data,
    generate_balanced_move_data,
    generate_progressive_difficulty_data,
    generate_comprehensive_move_data
)
from torch.utils.data import Dataset
import numpy as np
import torch
from tqdm import tqdm

# Cube moves
cube_moves = [
    "U", "D", "L", "R", "F", "B",
    "U'", "D'", "L'", "R'", "F'", "B'",
    "U2", "D2", "L2", "R2", "F2", "B2",
    "u", "d", "l", "r", "f", "b",
    "u'", "d'", "l'", "r'", "f'", "b'",
    "u2", "d2", "l2", "r2", "f2", "b2",
    "M", "E", "S", "x", "y", "z",
    "M'", "E'", "S'", "x'", "y'", "z'",
    "M2", "E2", "S2", "x2", "y2", "z2"
]

# Cube colors (using different names to avoid conflicts)
cube_colors = ["WHITE", "YELLOW", "GREEN", "BLUE", "RED", "ORANGE"]

# Combine all tokens
all_tokens = cube_moves + cube_colors
VOCAB = {token: i for i, token in enumerate(all_tokens)}

# Create reverse mapping for colors to handle cube state parsing
COLOR_MAP = {"W": "WHITE", "Y": "YELLOW", "G": "GREEN", "B": "BLUE", "R": "RED", "O": "ORANGE"}

class PreTrainDataset(Dataset):
    """
    FIXED Dataset for pre-training the model.
    
    This dataset pre-generates ALL data during initialization to ensure
    consistent data across training, evaluation, and testing.

    This fixes the critical bug where data was regenerated on every __getitem__ call.
    """
    def __init__(self, length: int, seed: int = 42):
        self.len = length
        self.seed = seed
        
        print(f"Generating {length:,} fixed samples (seed: {seed})...")
        
        # Set seed for reproducibility
        np.random.seed(seed)
        
        # Pre-generate ALL data
        self.data = []
        
        # Generate in batches for memory efficiency
        batch_size = min(1000, length)
        
        with tqdm(total=length, desc="Generating fixed dataset") as pbar:
            for start_idx in range(0, length, batch_size):
                end_idx = min(start_idx + batch_size, length)
                batch_length = end_idx - start_idx
                
                # Generate raw data
                raw_batch = generate_random_single_move_data(batch_length)
                
                # Process and store each sample
                for initial_state, move, final_state in raw_batch:
                    # Process exactly like the original dataset
                    initial_tensor = np.array([VOCAB[COLOR_MAP[face]] for face in initial_state.split(" ")])
                    move_tensor = VOCAB[move]
                    final_tensor = np.array([VOCAB[COLOR_MAP[face]] for face in final_state.split(" ")])
                    
                    # Store as tensors
                    self.data.append((
                        torch.tensor(initial_tensor, dtype=torch.long),
                        torch.tensor(move_tensor, dtype=torch.long),
                        torch.tensor(final_tensor, dtype=torch.long)
                    ))
                
                pbar.update(batch_length)
        
        print(f"Fixed dataset created with {len(self.data):,} consistent samples")
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # Return pre-generated data - always consistent!
        return self.data[idx]

class CurriculumDataset(Dataset):
    """
    Dataset for curriculum learning with progressive difficulty.
    """
    def __init__(self, length: int, difficulty_level: float = 0.5, use_balanced_moves: bool = True):
        self.len = length
        self.difficulty_level = difficulty_level
        self.use_balanced_moves = use_balanced_moves
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if self.use_balanced_moves:
            # Use balanced move generation for better learning
            max_scramble = max(1, min(25, int(1 + self.difficulty_level * 24)))
            initial_state, move, final_state = generate_balanced_move_data(1, max_scramble)[0]
        else:
            # Use progressive difficulty
            initial_state, move, final_state = generate_progressive_difficulty_data(1, self.difficulty_level)[0]
        
        # Map color letters to full color names to avoid conflicts
        initial_tensor = np.array([VOCAB[COLOR_MAP[face]] for face in initial_state.split(" ")])
        move_tensor = VOCAB[move]  # Single integer, not array
        final_tensor = np.array([VOCAB[COLOR_MAP[face]] for face in final_state.split(" ")])
        return torch.tensor(initial_tensor, dtype=torch.long), torch.tensor(move_tensor, dtype=torch.long), torch.tensor(final_tensor, dtype=torch.long)
    
    def update_difficulty(self, new_difficulty: float):
        """Update the difficulty level for curriculum learning."""
        self.difficulty_level = min(1.0, max(0.0, new_difficulty))

class AdvancedCurriculumDataset(Dataset):
    """
    FIXED Advanced curriculum dataset with multiple learning strategies.
    Pre-generates data with dynamic difficulty updates.
    """
    def __init__(self, length: int, min_scramble: int = 1, max_scramble: int = 25, 
                 balanced_moves: bool = True, difficulty_progression: str = "linear", seed: int = 42):
        self.len = length
        self.min_scramble = min_scramble
        self.target_max_scramble = max_scramble
        self.max_scramble = min_scramble + 1  # Start easy
        self.balanced_moves = balanced_moves
        self.difficulty_progression = difficulty_progression
        self.current_epoch = 0
        self.seed = seed
        
        print(f"Generating {length:,} fixed curriculum samples...")
        print(f"   Scramble range: {min_scramble}-{max_scramble}")
        print(f"   Balanced moves: {balanced_moves}")
        print(f"   Progression: {difficulty_progression}")
        
        # Set seed for reproducibility
        np.random.seed(seed)
        
        # Pre-generate ALL data with initial difficulty
        self._regenerate_data()
        
    def _regenerate_data(self):
        """Regenerate data with current difficulty level."""
        self.data = []
        
        # Generate in batches
        batch_size = min(1000, self.len)
        
        with tqdm(total=self.len, desc=f"Generating curriculum (max_scramble={self.max_scramble})") as pbar:
            for start_idx in range(0, self.len, batch_size):
                end_idx = min(start_idx + batch_size, self.len)
                batch_length = end_idx - start_idx
                
                # Generate curriculum data
                if self.balanced_moves:
                    raw_batch = generate_balanced_move_data(batch_length, max_scramble_length=self.max_scramble)
                else:
                    raw_batch = generate_curriculum_single_move_data(
                        batch_length, max_scramble_length=self.max_scramble, min_scramble_length=self.min_scramble
                    )
                
                # Process and store each sample
                for initial_state, move, final_state in raw_batch:
                    initial_tensor = np.array([VOCAB[COLOR_MAP[face]] for face in initial_state.split(" ")])
                    move_tensor = VOCAB[move]
                    final_tensor = np.array([VOCAB[COLOR_MAP[face]] for face in final_state.split(" ")])
                    
                    self.data.append((
                        torch.tensor(initial_tensor, dtype=torch.long),
                        torch.tensor(move_tensor, dtype=torch.long),
                        torch.tensor(final_tensor, dtype=torch.long)
                    ))
                
                pbar.update(batch_length)
        
        print(f"Fixed curriculum dataset updated with {len(self.data):,} samples")
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def update_curriculum(self, epoch: int, total_epochs: int):
        """Update curriculum parameters and regenerate data if difficulty changed."""
        self.current_epoch = epoch
        progress = epoch / total_epochs
        
        old_max_scramble = self.max_scramble
        
        if self.difficulty_progression == "linear":
            # Linear progression from min to max scramble length
            self.max_scramble = max(self.min_scramble + 1, self.min_scramble + int(progress * (self.target_max_scramble - self.min_scramble)))
        elif self.difficulty_progression == "exponential":
            # Exponential progression - stays easy longer, then ramps up quickly
            self.max_scramble = max(self.min_scramble + 1, self.min_scramble + int((progress ** 2) * (self.target_max_scramble - self.min_scramble)))
        elif self.difficulty_progression == "stepped":
            # Stepped progression - increases every few epochs
            steps = max(1, total_epochs // 10)  # 10 difficulty steps
            step_size = (self.target_max_scramble - self.min_scramble) / steps
            current_step = min(steps - 1, epoch // (total_epochs // steps))
            self.max_scramble = max(self.min_scramble + 1, self.min_scramble + int(current_step * step_size))
        
        # Only regenerate data if difficulty actually changed
        if self.max_scramble != old_max_scramble:
            print(f"🔄 Curriculum difficulty updated: {old_max_scramble} → {self.max_scramble}")
            self._regenerate_data()

class HybridDataset(Dataset):
    """
    Hybrid dataset that combines curriculum learning with standard moves
    to ensure comprehensive coverage of all possible Rubik's Cube moves.
    """
    def __init__(self, size, curriculum_ratio=0.6, standard_ratio=0.4, 
                 min_scramble=1, max_scramble=25, balanced_moves=True):
        self.size = size
        self.curriculum_ratio = curriculum_ratio
        self.standard_ratio = standard_ratio
        self.min_scramble = min_scramble
        self.max_scramble = max_scramble
        self.balanced_moves = balanced_moves
        
        # Calculate sizes for each dataset type
        self.curriculum_size = int(size * curriculum_ratio)
        self.standard_size = size - self.curriculum_size
        
        # Create the component datasets
        self.curriculum_data = AdvancedCurriculumDataset(
            self.curriculum_size,
            min_scramble=min_scramble,
            max_scramble=max_scramble,
            balanced_moves=balanced_moves,
            difficulty_progression="exponential"
        )
        
        self.standard_data = PreTrainDataset(self.standard_size)
        
        print(f"🔀 Hybrid Dataset Created:")
        print(f"   - Curriculum data: {self.curriculum_size:,} samples")
        print(f"   - Standard data: {self.standard_size:,} samples")
        print(f"   - Total: {size:,} samples")
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if idx < self.curriculum_size:
            return self.curriculum_data[idx]
        else:
            return self.standard_data[idx - self.curriculum_size]
    
    def update_curriculum(self, epoch, total_epochs):
        """Update curriculum difficulty for the curriculum portion."""
        if hasattr(self.curriculum_data, 'update_curriculum'):
            self.curriculum_data.update_curriculum(epoch, total_epochs)

class ComprehensiveMoveDataset(Dataset):
    """
    Dataset specifically designed to ensure comprehensive coverage of ALL possible moves.
    This dataset guarantees equal representation of every move type.
    """
    def __init__(self, size, min_scramble=1, max_scramble=25):
        self.size = size
        self.min_scramble = min_scramble
        self.max_scramble = max_scramble
        
        # Define ALL possible move categories with equal representation
        self.move_categories = {
            # Single layer face turns
            'face_turns': ['U', 'D', 'L', 'R', 'F', 'B'],
            'face_turns_inverse': ['U\'', 'D\'', 'L\'', 'R\'', 'F\'', 'B\''],
            'face_turns_double': ['U2', 'D2', 'L2', 'R2', 'F2', 'B2'],
            
            # Wide moves (two layers)
            'wide_turns': ['u', 'd', 'l', 'r', 'f', 'b'],
            'wide_turns_inverse': ['u\'', 'd\'', 'l\'', 'r\'', 'f\'', 'b\''],
            'wide_turns_double': ['u2', 'd2', 'l2', 'r2', 'f2', 'b2'],
            
            # Middle slice moves
            'slice_moves': ['E', 'M', 'S'],
            'slice_moves_inverse': ['E\'', 'M\'', 'S\''],
            'slice_moves_double': ['E2', 'M2', 'S2'],
            
            # Cube rotations
            'rotations': ['x', 'y', 'z'],
            'rotations_inverse': ['x\'', 'y\'', 'z\''],
            'rotations_double': ['x2', 'y2', 'z2']
        }
        
        # Calculate samples per category to ensure equal representation
        total_categories = len(self.move_categories)
        samples_per_category = size // total_categories
        self.samples_per_category = samples_per_category
        
        print(f"Comprehensive Move Dataset Created:")
        print(f"   - Total samples: {size:,}")
        print(f"   - Categories: {total_categories}")
        print(f"   - Samples per category: {samples_per_category:,}")
        print(f"   - Move types covered: {sum(len(moves) for moves in self.move_categories.values())}")
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Determine which category this sample belongs to
        category_idx = idx // self.samples_per_category
        category_name = list(self.move_categories.keys())[category_idx]
        moves_in_category = self.move_categories[category_name]
        
        # Generate data with focus on this move category
        initial_state, move, final_state = generate_comprehensive_move_data(
            batch_size=1,
            target_moves=moves_in_category,
            min_scramble=self.min_scramble,
            max_scramble=self.max_scramble,
            vocab=VOCAB,
            color_map=COLOR_MAP
        )
        
        return initial_state[0], move[0], final_state[0]
    
if __name__ == "__main__":
    dataset = PreTrainDataset(100)
    print(dataset[0])
    print(dataset[1])
    print(dataset[2])
    print(dataset[3])
    print(dataset[4])
    print(dataset[5])
    print(dataset[6])
    print(dataset[7])
    print(dataset[8])
    print(dataset[9])