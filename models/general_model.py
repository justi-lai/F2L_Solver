import torch
import torch.nn as nn
from mamba_ssm import Mamba

class TRMRefiner(nn.Module):
    """
    Tiny Recursive Model (TRM) Block.
    Iteratively refines the spatial embedding to resolve ambiguities (like 'Lookahead').
    """
    def __init__(self, d_model, loops=4):
        super().__init__()
        self.loops = loops
        self.d_model = d_model
        
        # The "Thinking" Cell - A simple gated recurrent unit is excellent for recursion
        self.recurrence = nn.GRUCell(d_model, d_model)
        
        # Refinement Projector
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        # x shape: [Batch, Seq_Len, d_model]
        # We treat the sequence as a batch of independent states for spatial refinement
        batch, seq, dim = x.shape
        x_flat = x.view(-1, dim)  # Flatten to [Batch*Seq, Dim]
        
        # Initialize 'Memory' (z)
        z = torch.zeros_like(x_flat)
        
        # The Recursive Loop (Thinking Depth)
        for _ in range(self.loops):
            # 1. Recur: Update memory based on input state
            z = self.recurrence(x_flat, z)
            
            # 2. Refine: Add residual information back to input
            delta = self.projection(z)
            x_flat = x_flat + delta # Residual connection
            
        # Reshape back to sequence
        x_refined = self.norm(x_flat).view(batch, seq, dim)
        return x_refined

class MambaBlock(nn.Module):
    """
    Standard Mamba Block wrapping the CUDA kernel.
    """
    def __init__(self, d_model):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,    # Model dimension d_model
            d_state=16,         # SSM state expansion factor
            d_conv=4,           # Local convolution width
            expand=2,           # Block expansion factor
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Mamba requires specific shapes, handles normalization internally usually
        # but adding Pre-Norm is standard stability practice
        return self.norm(x + self.mamba(x))

class CubeSolver(nn.Module):
    def __init__(
        self, 
        input_dim=480,      # 20 pieces * 24 positions
        d_model=512,        # Latent dimension
        n_mamba_layers=8,   # Depth of backbone
        trm_loops=4,        # Depth of recursive eye
        vocab_size=21       # 12 Moves + 6 Rotations + 3 Middle
    ):
        super().__init__()
        
        # 1. Feature Extraction (DeepCube Style)
        # We use a simple MLP encoder instead of ResNet for speed, 
        # as TRM handles the spatial complexity.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 2. TRM Refiner (The "Eye")
        self.trm = TRMRefiner(d_model, loops=trm_loops)
        
        # 3. Mamba Backbone (The "Muscle")
        self.layers = nn.ModuleList([
            MambaBlock(d_model) for _ in range(n_mamba_layers)
        ])
        
        # 4. Action Head
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        Args:
            x: [Batch, Seq_Len, 480] - The sequence of cube states (trajectory)
        Returns:
            logits: [Batch, Seq_Len, Vocab_Size] - Prediction for NEXT move
        """
        # Encoder
        x = self.encoder(x)
        
        # Refiner (Think about the pattern)
        x = self.trm(x)
        
        # Backbone (Process sequence history)
        for layer in self.layers:
            x = layer(x)
            
        # Predict
        logits = self.head(x)
        return logits

# --- Usage Example ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize Model
    model = CubeSolver(d_model=512, n_mamba_layers=8, trm_loops=4).to(device)
    
    # Fake Batch of Data (Batch=32, Seq_Len=10 moves so far, Input=480)
    # This represents 32 different cubes, each having a history of 10 moves.
    dummy_input = torch.randn(32, 10, 480).to(device)
    
    # Forward Pass
    logits = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {logits.shape}") # Expect [32, 10, 19]
    print("Model initialized successfully!")