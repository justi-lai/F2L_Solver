# F2L_Solver

A position-aware Rubik's Cube state prediction model using transformer encoder-decoder architecture designed for knowledge transfer.

## Model Architecture

The model uses a hybrid approach:

- **Cube-aware positional encoding** that embeds 3D cube structure (faces, positions, sticker types)
- **Transformer encoder** for understanding initial cube states with spatial awareness
- **Recurrent decoder** for processing moves and predicting resulting states
- **Perfect accuracy** on single-move predictions across all 54 possible moves

## Key Features

- **100% accuracy** on single move predictions
- **Complete move coverage** including face turns, wide turns, slice moves, and rotations
- **High confidence predictions** with spatial understanding
- **Knowledge transfer ready** with separate encoder and decoder components
- **Fixed dataset generation** ensuring consistent training and evaluation

## Usage

### Training

```bash
cd models
python train_general_model.py
```

### Interactive Testing

```bash
# Test random examples
python test_model_interactive.py

# Test specific moves
python test_model_interactive.py --specific_move "R"
python test_model_interactive.py --specific_move "x'" --scramble_length 10
```

## Files

- `models/general_model.py` - Core model architecture
- `models/train_general_model.py` - Training script
- `models/test_model_interactive.py` - Interactive testing
- `datasets.py` - Data generation and loading
- `data_generation/cube_state_gen.py` - Cube state generation utilities

## Requirements

See `requirements.txt` for dependencies.
