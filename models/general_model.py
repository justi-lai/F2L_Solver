import torch
import torch.nn as nn
import math

# --- Position-Aware Components ---

class CubePositionalEncoding(nn.Module):
    """
    Position-aware encoding that understands cube structure:
    - 6 faces with 9 positions each
    - Face adjacencies and relationships
    - 3D spatial understanding
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # Cube structure understanding
        # Calculate embedding dimensions that sum to d_model
        face_dim = d_model // 4
        pos_dim = d_model // 4  
        type_dim = d_model // 4
        remaining_dim = d_model - (face_dim + pos_dim + type_dim)
        
        # Face embeddings (6 faces: U, L, F, R, B, D)
        self.face_embeddings = nn.Embedding(6, face_dim)
        
        # Position within face (9 positions: 0-8)
        self.position_embeddings = nn.Embedding(9, pos_dim)
        
        # Sticker type embeddings (corner=0, edge=1, center=2)
        self.type_embeddings = nn.Embedding(3, type_dim)
        
        # Additional structural embedding for remaining dimensions
        self.structural_embedding = nn.Parameter(torch.randn(54, remaining_dim))
        
        # Linear layer to combine all embeddings
        combined_dim = face_dim + pos_dim + type_dim + remaining_dim
        self.combine = nn.Linear(combined_dim, d_model)
        
        # Register cube structure as buffers
        self.register_buffer('cube_faces', self._build_face_mapping())
        self.register_buffer('cube_positions', self._build_position_mapping())
        self.register_buffer('cube_types', self._build_type_mapping())
        
    def _build_face_mapping(self):
        """Map each of 54 stickers to its face (0-5)"""
        # Face order: U(0), L(1), F(2), R(3), B(4), D(5)
        faces = torch.tensor([
            [0] * 9,  # Top face (U): stickers 0-8
            [1] * 9,  # Left face (L): stickers 9-17
            [2] * 9,  # Front face (F): stickers 18-26
            [3] * 9,  # Right face (R): stickers 27-35
            [4] * 9,  # Back face (B): stickers 36-44
            [5] * 9   # Down face (D): stickers 45-53
        ]).flatten()
        return faces
        
    def _build_position_mapping(self):
        """Map each sticker to its position within face (0-8)"""
        # Position layout within each face:
        # 0 1 2
        # 3 4 5
        # 6 7 8
        positions = torch.tensor(list(range(9)) * 6)
        return positions
        
    def _build_type_mapping(self):
        """Map each sticker to its type: corner=0, edge=1, center=2"""
        # Pattern for each face: corners, edges, center
        face_pattern = torch.tensor([0, 1, 0, 1, 2, 1, 0, 1, 0])  # corner, edge, corner, edge, center, edge, corner, edge, corner
        sticker_types = face_pattern.repeat(6)
        return sticker_types

    def forward(self, x):
        # x shape: (batch_size, 54, d_model)
        batch_size = x.size(0)
        
        # Get structural embeddings for all 54 positions
        face_embs = self.face_embeddings(self.cube_faces)      # (54, face_dim)
        pos_embs = self.position_embeddings(self.cube_positions)  # (54, pos_dim)
        type_embs = self.type_embeddings(self.cube_types)      # (54, type_dim)
        
        # Combine structural information
        structure_emb = torch.cat([face_embs, pos_embs, type_embs, self.structural_embedding], dim=-1)  # (54, combined_dim)
        structure_emb = self.combine(structure_emb)  # (54, d_model)
        
        # Add to input embeddings
        structure_emb = structure_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, 54, d_model)
        enhanced_x = x + structure_emb
        
        return self.dropout(enhanced_x)

class SequentialPositionalEncoding(nn.Module):
    """Standard positional encoding for sequences (used in encoder)"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embedding_dim)
        x = x + self.pe[:x.size(1)].permute(1, 0, 2)
        return self.dropout(x)

# --- Encoder-Decoder Architecture ---

class CubeStateEncoder(nn.Module):
    """
    ENCODER: Understands initial cube state with spatial awareness.
    This will be reusable for knowledge transfer - experts/analyzers.
    """
    def __init__(self, vocab_size, d_model, nhead, d_hid, nlayers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Color embedding for cube stickers
        self.color_embedding = nn.Embedding(vocab_size, d_model)
        
        # Position-aware encoding (understands cube structure)
        self.cube_pos_encoder = CubePositionalEncoding(d_model, dropout)
        
        # Transformer layers for understanding state relationships
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        
        # State summarization (for decoder initialization)
        self.state_summary = nn.Linear(d_model, d_model)
        
    def forward(self, cube_state):
        # cube_state shape: (batch_size, 54) - color indices for each sticker
        
        # 1. Embed colors
        color_embs = self.color_embedding(cube_state) * math.sqrt(self.d_model)  # (batch_size, 54, d_model)
        
        # 2. Add cube-aware positional encoding
        positioned_embs = self.cube_pos_encoder(color_embs)  # (batch_size, 54, d_model)
        
        # 3. Process through transformer to understand spatial relationships
        state_repr = self.transformer_encoder(positioned_embs)  # (batch_size, 54, d_model)
        
        # 4. Create global state summary for decoder initialization
        state_summary = self.state_summary(state_repr.mean(dim=1))  # (batch_size, d_model)
        
        return {
            'sticker_representations': state_repr,      # Per-sticker understanding
            'global_state': state_summary,              # Overall cube state
            'spatial_embeddings': positioned_embs       # For spatial reasoning
        }

class RecurrentDecoder(nn.Module):
    """
    DECODER: Recurrent processing of moves to predict state changes.
    Parts of this will be reusable for knowledge transfer.
    """
    def __init__(self, vocab_size, d_model, n_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Move embedding
        self.move_embedding = nn.Embedding(vocab_size, d_model)
        
        # Recurrent layers for processing move sequences
        self.rnn_layers = nn.ModuleList([
            SSMBlock(d_model) for _ in range(n_layers)
        ])
        
        # State transformation layers
        self.state_transform = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Output projection to colors
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, encoder_output, moves, max_steps=1):
        """
        Process moves recurrently.
        
        Args:
            encoder_output: Dict from encoder with state representations
            moves: (batch_size, seq_len) or (batch_size,) for single move
            max_steps: Maximum number of recurrent steps (1 for current training)
        """
        sticker_repr = encoder_output['sticker_representations']  # (batch_size, 54, d_model)
        
        # Handle single move (current training) vs move sequences (future)
        if moves.dim() == 1:
            moves = moves.unsqueeze(1)  # (batch_size, 1)
        
        batch_size, seq_len = moves.shape
        current_state = sticker_repr
        
        # Process each move in sequence
        for step in range(min(seq_len, max_steps)):
            # Get current move
            current_move = moves[:, step]  # (batch_size,)
            move_emb = self.move_embedding(current_move)  # (batch_size, d_model)
            
            # Broadcast move to all stickers
            move_broadcast = move_emb.unsqueeze(1).expand(-1, 54, -1)  # (batch_size, 54, d_model)
            
            # Apply recurrent transformation
            for rnn_layer in self.rnn_layers:
                current_state, _ = rnn_layer(move_broadcast, current_state)
            
            # Apply state transformation
            current_state = self.state_transform(current_state)
        
        # Project to color predictions
        output_logits = self.output_projection(current_state)  # (batch_size, 54, vocab_size)
        
        return output_logits

class SSMBlock(nn.Module):
    """Enhanced SSM layer with better state transitions"""
    def __init__(self, d_model):
        super().__init__()
        # State space matrices
        self.A = nn.Linear(d_model, d_model)  # State transition
        self.B = nn.Linear(d_model, d_model)  # Input influence  
        self.C = nn.Linear(d_model, d_model)  # Output transformation
        self.D = nn.Linear(d_model, d_model)  # Direct path (skip connection)
        
        # Normalization and activation
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, move_input, state_hidden):
        # move_input: (batch_size, 54, d_model) - move information
        # state_hidden: (batch_size, 54, d_model) - current state
        
        # State space update: x_{t+1} = Ax_t + Bu_t
        state_update = self.A(state_hidden) + self.B(move_input)
        state_update = self.activation(state_update)
        state_update = self.norm1(state_update + state_hidden)  # Residual
        
        # Output: y_t = Cx_{t+1} + Du_t
        output = self.C(state_update) + self.D(move_input)
        output = self.dropout(output)
        output = self.norm2(output + state_update)  # Another residual
        
        return output, state_update

# --- Complete Position-Aware Encoder-Decoder Model ---

class PositionAwareFoundationalModel(nn.Module):
    """
    Complete encoder-decoder model with spatial cube understanding.
    
    Architecture:
    - Encoder: Understands initial cube state with 3D spatial awareness
    - Decoder: Recurrently processes moves to predict state changes
    
    Designed for knowledge transfer:
    - Encoder can be reused for state analysis
    - Decoder components can be reused for move prediction
    """
    def __init__(self, vocab_size, d_model=256, nhead=8, d_hid=1024, 
                 n_encoder_layers=4, n_decoder_layers=3, dropout=0.1):
        super().__init__()
        
        # Encoder: State understanding with spatial awareness
        self.encoder = CubeStateEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            d_hid=d_hid,
            nlayers=n_encoder_layers,
            dropout=dropout
        )
        
        # Decoder: Recurrent move processing
        self.decoder = RecurrentDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_decoder_layers,
            dropout=dropout
        )
        
        self.d_model = d_model
        
    def forward(self, initial_state, move):
        """
        Forward pass for single move (current training setup).
        
        Args:
            initial_state: (batch_size, 54) - cube state as color indices
            move: (batch_size,) - single move index
            
        Returns:
            output_logits: (batch_size, 54, vocab_size) - predicted final state
        """
        # Encode initial state with spatial understanding
        encoder_output = self.encoder(initial_state)
        
        # Decode move to predict final state
        output_logits = self.decoder(encoder_output, move, max_steps=1)
        
        return output_logits
    
    def forward_sequence(self, initial_state, move_sequence):
        """
        Forward pass for move sequences (future capability).
        
        Args:
            initial_state: (batch_size, 54) - cube state as color indices  
            move_sequence: (batch_size, seq_len) - sequence of moves
            
        Returns:
            output_logits: (batch_size, 54, vocab_size) - predicted final state
        """
        encoder_output = self.encoder(initial_state)
        output_logits = self.decoder(encoder_output, move_sequence, max_steps=move_sequence.size(1))
        return output_logits
    
    def encode_state(self, cube_state):
        """
        Extract state representations for knowledge transfer.
        
        Args:
            cube_state: (batch_size, 54) - cube state as color indices
            
        Returns:
            Dict with state representations for downstream tasks
        """
        return self.encoder(cube_state)
    
    def get_encoder_for_transfer(self):
        """Get encoder for knowledge transfer to experts/analyzers"""
        return self.encoder
    
    def get_decoder_components_for_transfer(self):
        """Get decoder components for knowledge transfer"""
        return {
            'move_embedding': self.decoder.move_embedding,
            'rnn_layers': self.decoder.rnn_layers,
            'state_transform': self.decoder.state_transform
        }

# Legacy model for backward compatibility
FoundationalModel = PositionAwareFoundationalModel

# Model architecture complete - training functions moved to train_general_model.py