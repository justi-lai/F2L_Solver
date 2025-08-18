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


class MultiMoveDataset(Dataset):
    """
    Dataset for multi-move sequence training.
    
    Each sample contains:
    - initial_state: 54 sticker colors (tokenized)
    - move_sequence: sequence of moves (tokenized)
    - final_state: resulting 54 sticker colors after all moves (tokenized)
    """
    
    def __init__(self, length: int, min_seq_length: int = 2, max_seq_length: int = 8, seed: int = 42):
        """
        Initialize multi-move dataset.
        
        Args:
            length: Number of samples to generate
            min_seq_length: Minimum number of moves in sequence
            max_seq_length: Maximum number of moves in sequence
            seed: Random seed for reproducibility
        """
        self.len = length
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seed = seed
        
        print(f"Generating {length:,} multi-move samples (sequences: {min_seq_length}-{max_seq_length} moves, seed: {seed})...")
        
        # Set seed for reproducibility
        np.random.seed(seed)
        
        # Get move tokens (exclude color tokens)
        all_tokens = list(VOCAB.keys())
        self.move_tokens = [token for token in all_tokens if token not in ['WHITE', 'YELLOW', 'GREEN', 'BLUE', 'RED', 'ORANGE']]
        
        # Pre-generate ALL data
        self.data = []
        
        # Generate in batches for memory efficiency
        batch_size = min(100, length)
        
        with tqdm(total=length, desc="Generating multi-move dataset") as pbar:
            for start_idx in range(0, length, batch_size):
                end_idx = min(start_idx + batch_size, length)
                batch_length = end_idx - start_idx
                
                # Generate raw data batch
                raw_batch = self._generate_batch(batch_length)
                
                # Process and store each sample
                for initial_state, move_sequence, final_state in raw_batch:
                    # Tokenize initial state
                    initial_tensor = np.array([VOCAB[COLOR_MAP[face]] for face in initial_state.split(" ")])
                    
                    # Tokenize move sequence
                    move_tensors = np.array([VOCAB[move] for move in move_sequence])
                    
                    # Tokenize final state
                    final_tensor = np.array([VOCAB[COLOR_MAP[face]] for face in final_state.split(" ")])
                    
                    # Store as tensors
                    self.data.append((
                        torch.tensor(initial_tensor, dtype=torch.long),
                        torch.tensor(move_tensors, dtype=torch.long),
                        torch.tensor(final_tensor, dtype=torch.long)
                    ))
                
                pbar.update(batch_length)
        
        print(f"Multi-move dataset created with {len(self.data):,} consistent samples")
        
        # Show sequence length distribution
        seq_lengths = [len(self.data[i][1]) for i in range(min(100, len(self.data)))]
        print(f"   Sample sequence lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, avg={np.mean(seq_lengths):.1f}")
    
    def _generate_batch(self, batch_size: int):
        """Generate a batch of multi-move samples."""
        from cuber.cuber import Cube
        from data_generation.cube_state_gen import generate_random_scramble
        
        batch_data = []
        
        for _ in range(batch_size):
            # Random sequence length
            seq_length = np.random.randint(self.min_seq_length, self.max_seq_length + 1)
            
            # Generate initial scramble
            initial_scramble_length = np.random.randint(5, 20)
            initial_scramble = generate_random_scramble(initial_scramble_length)
            
            # Create cube and apply initial scramble
            cube = Cube()
            cube.turn(initial_scramble)
            initial_state = cube.get_faces_str("ULFRBD")
            
            # Generate and apply move sequence
            move_sequence = []
            for _ in range(seq_length):
                move = np.random.choice(self.move_tokens)
                move_sequence.append(move)
                cube.turn(move)
            
            final_state = cube.get_faces_str("ULFRBD")
            
            batch_data.append((initial_state, move_sequence, final_state))
        
        return batch_data
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_max_sequence_length(self):
        """Get the maximum sequence length in the dataset."""
        return max(len(item[1]) for item in self.data)


class CurriculumMultiMoveDataset(Dataset):
    """
    Curriculum learning version of multi-move dataset.
    
    Starts with shorter sequences and gradually increases difficulty.
    Uses smarter curriculum progression for better learning.
    """
    
    def __init__(self, length: int, initial_max_length: int = 2, final_max_length: int = 8, seed: int = 42):
        """
        Initialize curriculum multi-move dataset.
        
        Args:
            length: Number of samples to generate
            initial_max_length: Starting maximum sequence length
            final_max_length: Final maximum sequence length
            seed: Random seed for reproducibility
        """
        self.len = length
        self.initial_max_length = initial_max_length
        self.final_max_length = final_max_length
        self.current_max_length = initial_max_length
        self.seed = seed
        
        print(f"Creating curriculum multi-move dataset:")
        print(f"   Samples: {length:,}")
        print(f"   Curriculum: {initial_max_length} → {final_max_length} moves")
        print(f"   Seed: {seed}")
        
        # Get move tokens
        all_tokens = list(VOCAB.keys())
        self.move_tokens = [token for token in all_tokens if token not in ['WHITE', 'YELLOW', 'GREEN', 'BLUE', 'RED', 'ORANGE']]
        
        # Generate initial dataset
        self._regenerate_data()
    
    def _regenerate_data(self):
        """Regenerate data based on current curriculum difficulty."""
        np.random.seed(self.seed + hash(str(self.current_max_length)) % 10000)
        
        print(f"Generating data with max sequence length: {self.current_max_length}")
        
        self.data = []
        
        with tqdm(total=self.len, desc=f"Curriculum generation (max_len={self.current_max_length})") as pbar:
            batch_size = 100
            for start_idx in range(0, self.len, batch_size):
                end_idx = min(start_idx + batch_size, self.len)
                batch_length = end_idx - start_idx
                
                raw_batch = self._generate_curriculum_batch(batch_length)
                
                for initial_state, move_sequence, final_state in raw_batch:
                    initial_tensor = np.array([VOCAB[COLOR_MAP[face]] for face in initial_state.split(" ")])
                    move_tensors = np.array([VOCAB[move] for move in move_sequence])
                    final_tensor = np.array([VOCAB[COLOR_MAP[face]] for face in final_state.split(" ")])
                    
                    self.data.append((
                        torch.tensor(initial_tensor, dtype=torch.long),
                        torch.tensor(move_tensors, dtype=torch.long),
                        torch.tensor(final_tensor, dtype=torch.long)
                    ))
                
                pbar.update(batch_length)
    
    def _generate_curriculum_batch(self, batch_size: int):
        """Generate batch based on current curriculum level with intermediate states."""
        from cuber.cuber import Cube
        from data_generation.cube_state_gen import generate_random_scramble
        
        batch_data = []
        
        for _ in range(batch_size):
            # ENHANCED: Curriculum with knowledge retention
            # Sample from all difficulties learned so far, with more weight on current level
            seq_length = self._sample_sequence_length_with_retention()
            
            # Generate initial state
            initial_scramble_length = np.random.randint(5, 15)
            initial_scramble = generate_random_scramble(initial_scramble_length)
            
            cube = Cube()
            cube.turn(initial_scramble)
            initial_state = cube.get_faces_str("ULFRBD")
            
            # Generate move sequence (back to fast generation)
            move_sequence = []
            for _ in range(seq_length):
                move = np.random.choice(self.move_tokens)
                move_sequence.append(move)
                cube.turn(move)
            
            final_state = cube.get_faces_str("ULFRBD")
            # Store simple format for speed
            batch_data.append((initial_state, move_sequence, final_state))
        
        return batch_data
    
    def _sample_sequence_length_with_retention(self):
        """
        Sample sequence length with knowledge retention and diversity.
        
        Distribution ensures diversity across all learned sequence lengths:
        - 40% current max difficulty
        - 25% immediate previous difficulties 
        - 20% older difficulties
        - 15% uniform diversity across all lengths
        """
        if self.current_max_length == 1:
            return 1
        
        rand_val = np.random.random()
        
        if rand_val < 0.4:
            # 40% current max difficulty
            return self.current_max_length
        elif rand_val < 0.65:
            # 25% immediate previous difficulties (current-1 to current-2)
            if self.current_max_length >= 2:
                lower_bound = max(1, self.current_max_length - 2)
                upper_bound = self.current_max_length
                return np.random.randint(lower_bound, upper_bound)
            else:
                return 1
        elif rand_val < 0.85:
            # 20% older difficulties (1 to current-3)
            if self.current_max_length >= 4:
                return np.random.randint(1, max(2, self.current_max_length - 2))
            elif self.current_max_length >= 2:
                return 1
            else:
                return 1
        else:
            # 15% uniform diversity - ensures all sequence lengths get representation
            return np.random.randint(1, self.current_max_length + 1)
    
    def update_curriculum(self, progress_ratio: float):
        """
        Update curriculum difficulty based on training progress.
        
        Args:
            progress_ratio: Training progress (0.0 to 1.0)
        """
        # IMPROVED: Smoother curriculum progression
        new_max_length = int(self.initial_max_length + 
                           (self.final_max_length - self.initial_max_length) * progress_ratio)
        
        if new_max_length != self.current_max_length:
            self.current_max_length = new_max_length
            self._regenerate_data()
            return True
        return False
    
    def get_current_curriculum_range(self):
        """Get the current sequence length range for eval dataset alignment."""
        if self.current_max_length == 1:
            return 1, 1
        else:
            return max(1, self.current_max_length - 1), self.current_max_length
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_max_sequence_length(self):
        """Get the current maximum sequence length."""
        return self.current_max_length


class AlignedEvalMultiMoveDataset(Dataset):
    """
    Evaluation dataset that aligns with the curriculum training dataset.
    
    This dataset dynamically adjusts its sequence length distribution
    to match the current curriculum level of the training dataset.
    """
    
    def __init__(self, length: int, train_dataset=None, seed: int = 999):
        """
        Initialize aligned evaluation dataset.
        
        Args:
            length: Number of samples to generate
            train_dataset: Training dataset to align with (optional, can align later)
            seed: Random seed for reproducibility
        """
        self.len = length
        self.seed = seed
        self.current_min_length = 1
        self.current_max_length = 1
        
        print(f"Creating aligned evaluation multi-move dataset:")
        print(f"   Samples: {length:,}")
        print(f"   Will align with training curriculum dynamically")
        print(f"   Seed: {seed}")
        
        # Get move tokens
        all_tokens = list(VOCAB.keys())
        self.move_tokens = [token for token in all_tokens if token not in ['WHITE', 'YELLOW', 'GREEN', 'BLUE', 'RED', 'ORANGE']]
        
        # Initialize data
        self.data = []
        
        # If training dataset provided, align immediately, otherwise use default
        if train_dataset is not None and hasattr(train_dataset, 'get_current_curriculum_range'):
            self.current_min_length, self.current_max_length = train_dataset.get_current_curriculum_range()
        
        self._regenerate_data()
    
    def _regenerate_data(self):
        """Regenerate data based on current curriculum alignment."""
        np.random.seed(self.seed + hash(f"{self.current_min_length}_{self.current_max_length}") % 10000)
        
        print(f"Regenerating eval data: sequence lengths {self.current_min_length}-{self.current_max_length}")
        
        self.data = []
        
        with tqdm(total=self.len, desc=f"Eval generation (len={self.current_min_length}-{self.current_max_length})") as pbar:
            batch_size = 100
            for start_idx in range(0, self.len, batch_size):
                end_idx = min(start_idx + batch_size, self.len)
                batch_length = end_idx - start_idx
                
                raw_batch = self._generate_aligned_batch(batch_length)
                
                for initial_state, move_sequence, final_state in raw_batch:
                    initial_tensor = np.array([VOCAB[COLOR_MAP[face]] for face in initial_state.split(" ")])
                    move_tensors = np.array([VOCAB[move] for move in move_sequence])
                    final_tensor = np.array([VOCAB[COLOR_MAP[face]] for face in final_state.split(" ")])
                    
                    self.data.append((
                        torch.tensor(initial_tensor, dtype=torch.long),
                        torch.tensor(move_tensors, dtype=torch.long),
                        torch.tensor(final_tensor, dtype=torch.long)
                    ))
                
                pbar.update(batch_length)
    
    def _generate_aligned_batch(self, batch_size: int):
        """Generate batch aligned with current curriculum level."""
        from cuber.cuber import Cube
        from data_generation.cube_state_gen import generate_random_scramble
        
        batch_data = []
        
        for _ in range(batch_size):
            # Match the enhanced curriculum distribution with knowledge retention
            seq_length = self._sample_eval_sequence_length()
            
            # Generate initial state
            initial_scramble_length = np.random.randint(5, 15)
            initial_scramble = generate_random_scramble(initial_scramble_length)
            
            cube = Cube()
            cube.turn(initial_scramble)
            initial_state = cube.get_faces_str("ULFRBD")
            
            # Generate move sequence (back to fast generation)
            move_sequence = []
            for _ in range(seq_length):
                move = np.random.choice(self.move_tokens)
                move_sequence.append(move)
                cube.turn(move)
            
            final_state = cube.get_faces_str("ULFRBD")
            batch_data.append((initial_state, move_sequence, final_state))
        
        return batch_data
    
    def _sample_eval_sequence_length(self):
        """
        Sample sequence length for evaluation to match training distribution.
        Tests all difficulties learned so far with proper weighting.
        """
        if self.current_max_length == 1:
            return 1
        
        rand_val = np.random.random()
        
        if rand_val < 0.4:
            # 40% current max difficulty
            return self.current_max_length
        elif rand_val < 0.7:
            # 30% immediate previous difficulties
            if self.current_max_length >= 2:
                lower_bound = max(1, self.current_max_length - 2)
                upper_bound = self.current_max_length
                return np.random.randint(lower_bound, upper_bound)
            else:
                return 1
        else:
            # 30% uniform distribution over all learned difficulties
            return np.random.randint(1, self.current_max_length + 1)
    
    def align_with_curriculum(self, train_dataset):
        """
        Align this eval dataset with the current curriculum level.
        
        Args:
            train_dataset: The curriculum training dataset to align with
        """
        if hasattr(train_dataset, 'get_current_curriculum_range'):
            new_min, new_max = train_dataset.get_current_curriculum_range()
            
            if new_min != self.current_min_length or new_max != self.current_max_length:
                self.current_min_length = new_min
                self.current_max_length = new_max
                self._regenerate_data()
                return True
        return False
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_max_sequence_length(self):
        """Get the current maximum sequence length."""
        return self.current_max_length

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