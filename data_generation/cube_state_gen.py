from cuber.cuber import Cube, MOVES
from random import randint
import torch
import random
# VOCAB and COLOR_MAP will be passed as parameters to avoid circular imports

def generate_random_scramble(length: int) -> str:
    """
    Generates a random scramble of a given length.
    The scramble is a string of moves separated by spaces.

    Args:
        length (int): The length of the scramble.

    Returns:
        str: The scramble.
    """
    scramble = ""
    for _ in range(length):
        scramble += MOVES[randint(0, len(MOVES) - 1)] + " "
    return scramble.strip()

def _generate_random_move(scramble: str) -> tuple[str, str, str]:
    """
    Generates a random move from a given scramble.

    Args:
        scramble (str): The scramble.

    Returns:
        tuple[str, str, str]: The move.
    """
    cube = Cube()
    cube.turn(scramble)
    initial_state = cube.get_faces_str("ULFRBD")
    move = MOVES[randint(0, len(MOVES) - 1)]
    cube.turn(move)
    final_state = cube.get_faces_str("ULFRBD")
    return initial_state, move, final_state

def generate_random_single_move_data(batch_size: int) -> list[tuple[str, str, str]]:
    """
    Generates random single move prediction data.
    The data is a list of tuples, where each tuple contains a scramble, a cube state, and a single move.

    Args:
        batch_size (int): The number of data points to generate.

    Returns:
        list[tuple[str, str, str]]: The data.
    """
    data = []
    for _ in range(batch_size):
        initial_state, move, final_state = _generate_random_move(generate_random_scramble(randint(10, 20)))
        data.append((initial_state, move, final_state))
    return data

def generate_curriculum_single_move_data(batch_size: int, max_scramble_length: int = 20, min_scramble_length: int = 1) -> list[tuple[str, str, str]]:
    """
    Generates curriculum learning data with controlled scramble complexity.
    
    Args:
        batch_size (int): The number of data points to generate.
        max_scramble_length (int): Maximum scramble length for this batch.
        min_scramble_length (int): Minimum scramble length for this batch.
    
    Returns:
        list[tuple[str, str, str]]: The data with (initial_state, move, final_state).
    """
    data = []
    for _ in range(batch_size):
        # Generate scramble with length between min and max
        scramble_length = randint(min_scramble_length, max_scramble_length)
        scramble = generate_random_scramble(scramble_length)
        initial_state, move, final_state = _generate_random_move(scramble)
        data.append((initial_state, move, final_state))
    return data

def generate_balanced_move_data(batch_size: int, max_scramble_length: int = 20, move_groups: dict = None) -> list[tuple[str, str, str]]:
    """
    Generates data with balanced move types for better learning.
    
    Args:
        batch_size (int): The number of data points to generate.
        max_scramble_length (int): Maximum scramble length.
        move_groups (dict): Dictionary defining move groups and their weights.
    
    Returns:
        list[tuple[str, str, str]]: The data with balanced move distribution.
    """
    if move_groups is None:
        # Default move groups - COMPLETE coverage of all 54 moves
        move_groups = {
            'face_turns': ['U', 'D', 'L', 'R', 'F', 'B'],           # Basic face turns
            'face_primes': ["U'", "D'", "L'", "R'", "F'", "B'"],    # Prime moves  
            'face_doubles': ['U2', 'D2', 'L2', 'R2', 'F2', 'B2'],  # Double turns
            'wide_turns': ['u', 'd', 'l', 'r', 'f', 'b'],          # Wide turns
            'wide_primes': ["u'", "d'", "l'", "r'", "f'", "b'"],   # Wide turn primes
            'wide_doubles': ['u2', 'd2', 'l2', 'r2', 'f2', 'b2'],  # Wide turn doubles
            'slice_turns': ['M', 'E', 'S'],                         # Slice turns
            'slice_primes': ["M'", "E'", "S'"],                     # Slice primes
            'slice_doubles': ['M2', 'E2', 'S2'],                    # Slice doubles
            'rotations': ['x', 'y', 'z'],                           # Cube rotations
            'rotation_primes': ["x'", "y'", "z'"],                  # Rotation primes
            'rotation_doubles': ['x2', 'y2', 'z2']                  # Rotation doubles
        }
    
    data = []
    group_names = list(move_groups.keys())
    
    for i in range(batch_size):
        # Cycle through move groups to ensure balance
        group_name = group_names[i % len(group_names)]
        target_moves = move_groups[group_name]
        
        # Generate scramble
        scramble_length = randint(1, max_scramble_length)
        scramble = generate_random_scramble(scramble_length)
        
        # Apply scramble
        cube = Cube()
        cube.turn(scramble)
        initial_state = cube.get_faces_str("ULFRBD")
        
        # Choose a move from the target group
        move = target_moves[randint(0, len(target_moves) - 1)]
        cube.turn(move)
        final_state = cube.get_faces_str("ULFRBD")
        
        data.append((initial_state, move, final_state))
    
    return data

def generate_progressive_difficulty_data(batch_size: int, difficulty_level: float = 0.5) -> list[tuple[str, str, str]]:
    """
    Generates data with progressive difficulty based on a difficulty level.
    
    Args:
        batch_size (int): The number of data points to generate.
        difficulty_level (float): Difficulty level from 0.0 (easiest) to 1.0 (hardest).
    
    Returns:
        list[tuple[str, str, str]]: The data with appropriate difficulty.
    """
    data = []
    
    # Map difficulty to scramble length and move complexity
    min_scramble = max(1, int(difficulty_level * 3))  # 1-3 moves for easy
    max_scramble = min(1 + int(difficulty_level * 25), 25)  # 1-25 moves
    
    # Choose move types based on difficulty
    if difficulty_level < 0.3:
        # Easy: basic face turns only
        allowed_moves = ['U', 'D', 'L', 'R', 'F', 'B', "U'", "D'", "L'", "R'", "F'", "B'"]
    elif difficulty_level < 0.6:
        # Medium: add double turns
        allowed_moves = ['U', 'D', 'L', 'R', 'F', 'B', "U'", "D'", "L'", "R'", "F'", "B'", 
                        'U2', 'D2', 'L2', 'R2', 'F2', 'B2']
    else:
        # Hard: all moves including rotations and slices
        allowed_moves = MOVES
    
    for _ in range(batch_size):
        # Generate scramble
        scramble_length = randint(min_scramble, max_scramble)
        scramble = generate_random_scramble(scramble_length)
        
        # Apply scramble
        cube = Cube()
        cube.turn(scramble)
        initial_state = cube.get_faces_str("ULFRBD")
        
        # Choose move based on difficulty
        move = allowed_moves[randint(0, len(allowed_moves) - 1)]
        cube.turn(move)
        final_state = cube.get_faces_str("ULFRBD")
        
        data.append((initial_state, move, final_state))
    
    return data

def generate_comprehensive_move_data(batch_size, target_moves=None, min_scramble_length=1, max_scramble_length=25, vocab=None, color_map=None):
    """
    Generate data with comprehensive coverage of all move types.
    
    Args:
        batch_size: Number of samples to generate
        target_moves: List of specific moves to focus on (if None, uses all moves)
        min_scramble_length: Minimum scramble length
        max_scramble_length: Maximum scramble length
    
    Returns:
        tuple: (initial_states, moves, final_states)
    """
    # Define ALL possible moves if target_moves not specified
    if target_moves is None:
        target_moves = [
            # Single layer face turns
            'U', 'D', 'L', 'R', 'F', 'B',
            'U\'', 'D\'', 'L\'', 'R\'', 'F\'', 'B\'',
            'U2', 'D2', 'L2', 'R2', 'F2', 'B2',
            
            # Wide moves (two layers)
            'u', 'd', 'l', 'r', 'f', 'b',
            'u\'', 'd\'', 'l\'', 'r\'', 'f\'', 'b\'',
            'u2', 'd2', 'l2', 'r2', 'f2', 'b2',
            
            # Middle slice moves
            'E', 'M', 'S',
            'E\'', 'M\'', 'S\'',
            'E2', 'M2', 'S2',
            
            # Cube rotations
            'x', 'y', 'z',
            'x\'', 'y\'', 'z\'',
            'x2', 'y2', 'z2'
        ]
    
    initial_states = []
    moves = []
    final_states = []
    
    for _ in range(batch_size):
        # Create a solved cube
        cube = Cube()
        
        # Apply random scramble
        scramble_length = random.randint(min_scramble_length, max_scramble_length)
        scramble_moves = []
        
        for _ in range(scramble_length):
            # Use moves from target_moves if specified, otherwise random
            if target_moves:
                move = random.choice(target_moves)
            else:
                move = random.choice(MOVES)
            scramble_moves.append(move)
            cube(move)
        
        # Get initial state (after scramble)
        initial_state = cube.flat_str()
        
        # Choose the final move from target_moves
        final_move = random.choice(target_moves)
        
        # Apply the final move
        cube(final_move)
        final_state = cube.flat_str()
        
        # Convert to tensor format using the same approach as existing functions
        if vocab is None or color_map is None:
            # Use default values if not provided
            vocab = {"U": 0, "D": 1, "L": 2, "R": 3, "F": 4, "B": 5}
            color_map = {"U": "U", "D": "D", "L": "L", "R": "R", "F": "F", "B": "B"}
        
        initial_state_tensor = torch.tensor([vocab[color_map[face]] for face in initial_state.split(" ")], dtype=torch.long)
        move_tensor = torch.tensor([vocab[final_move]], dtype=torch.long)
        final_state_tensor = torch.tensor([vocab[color_map[face]] for face in final_state.split(" ")], dtype=torch.long)
        
        initial_states.append(initial_state_tensor)
        moves.append(move_tensor)
        final_states.append(final_state_tensor)
    
    return torch.stack(initial_states), torch.stack(moves), torch.stack(final_states)



if __name__ == "__main__":
    data = generate_random_single_move_data(10)
    for index, (initial_state, move, final_state) in enumerate(data):
        print(f"Initial state: {initial_state}")
        print(f"Move: {move}")
        print(f"Final state: {final_state}")
        print(f"Index: {index}")
        print()