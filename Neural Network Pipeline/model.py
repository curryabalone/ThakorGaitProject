"""
Neural Contact Force Estimation Model

Architecture:
- Temporal Encoder: 2 stacked LSTM layers (256 hidden units)
- Spatial Decoder: Shared MLP (128 → 64 → 1) per sphere

Input: Sequence of foot poses (batch_size, seq_len, input_dim)
Output: Contact forces for each sphere (batch_size, num_spheres)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from get_batch import get_batch


class TemporalEncoder(nn.Module):
    """
    LSTM-based encoder that processes motion history.
    
    Takes a sequence of foot poses and outputs a latent state
    summarizing the current gait situation.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # Input shape: (batch, seq, features)
            dropout=0.1 if num_layers > 1 else 0.0
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Motion sequence (batch_size, seq_len, input_dim)
        
        Returns:
            Latent state (batch_size, hidden_dim)
        """
        # LSTM returns: output (all timesteps), (h_n, c_n) (final states)
        _, (h_n, _) = self.lstm(x)
        
        # h_n shape: (num_layers, batch, hidden_dim)
        # Take the last layer's hidden state
        return h_n[-1]


class SpatialDecoder(nn.Module):
    """
    MLP decoder that predicts force for each contact sphere.
    
    Same MLP is applied to each sphere with different inputs:
    - Sphere position (x, y, z)
    - Sphere radius
    - Gait state from temporal encoder
    """
    
    def __init__(self, latent_dim: int = 256, sphere_feature_dim: int = 4):
        super().__init__()
        
        # Input: sphere features (4) + latent state (256) = 260
        input_dim = sphere_feature_dim + latent_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, sphere_features: torch.Tensor, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sphere_features: (batch_size, num_spheres, 4) - [x, y, z, radius]
            latent_state: (batch_size, latent_dim) - from temporal encoder
        
        Returns:
            Forces: (batch_size, num_spheres) - predicted force per sphere
        """
        batch_size, num_spheres, _ = sphere_features.shape
        
        # Expand latent state to match each sphere
        # (batch, latent_dim) -> (batch, num_spheres, latent_dim)
        latent_expanded = latent_state.unsqueeze(1).expand(-1, num_spheres, -1)
        
        # Concatenate sphere features with latent state
        # (batch, num_spheres, 4 + latent_dim)
        combined = torch.cat([sphere_features, latent_expanded], dim=-1)
        
        # Apply MLP to each sphere (batch processing)
        # Reshape to (batch * num_spheres, input_dim) for efficient processing
        combined_flat = combined.view(-1, combined.shape[-1])
        forces_flat = self.mlp(combined_flat)
        
        # Reshape back to (batch, num_spheres)
        forces = forces_flat.view(batch_size, num_spheres)
        
        # Softplus ensures non-negative forces
        return F.softplus(forces)


class ContactForceModel(nn.Module):
    """
    Full model combining temporal encoder and spatial decoder.
    
    Predicts contact forces for each sphere given motion history.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_spheres: int = 240,
        hidden_dim: int = 256,
        num_lstm_layers: int = 2
    ):
        super().__init__()
        
        self.num_spheres = num_spheres
        
        self.temporal_encoder = TemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_lstm_layers
        )
        
        self.spatial_decoder = SpatialDecoder(
            latent_dim=hidden_dim,
            sphere_feature_dim=4  # x, y, z, radius
        )
    
    def forward(
        self,
        motion_sequence: torch.Tensor,
        sphere_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            motion_sequence: (batch_size, seq_len, input_dim) - foot pose history
            sphere_features: (batch_size, num_spheres, 4) - sphere geometry
        
        Returns:
            forces: (batch_size, num_spheres) - predicted contact forces
        """
        # Step 1: Encode motion history into latent state
        latent_state = self.temporal_encoder(motion_sequence)
        
        # Step 2: Decode forces for each sphere
        forces = self.spatial_decoder(sphere_features, latent_state)
        
        return forces


def create_dummy_sphere_features(num_spheres: int = 240) -> np.ndarray:
    """
    Create placeholder sphere features for testing.
    In practice, these come from your foot mesh geometry.
    
    Returns:
        (num_spheres, 4) array of [x, y, z, radius]
    """
    # Distribute spheres roughly across foot dimensions
    # x: 0 to 0.25m (heel to toe)
    # y: -0.05 to 0.05m (medial-lateral)
    # z: -0.02 to 0.02m (plantar surface)
    
    np.random.seed(42)
    x = np.random.uniform(0, 0.25, num_spheres)
    y = np.random.uniform(-0.05, 0.05, num_spheres)
    z = np.random.uniform(-0.02, 0.02, num_spheres)
    radius = np.random.uniform(0.005, 0.015, num_spheres)
    
    return np.stack([x, y, z, radius], axis=1).astype(np.float32)


if __name__ == "__main__":
    # Load a batch of motion data
    batch = get_batch(batch_size=10)  # (seq_len, num_features)
    print(f"Motion batch shape: {batch.shape}")
    
    seq_len, input_dim = batch.shape
    num_spheres = 240
    
    # Create model
    model = ContactForceModel(
        input_dim=input_dim,
        num_spheres=num_spheres,
        hidden_dim=256,
        num_lstm_layers=2
    )
    
    # Prepare inputs
    # Add batch dimension: (1, seq_len, input_dim)
    motion_tensor = torch.tensor(batch, dtype=torch.float32).unsqueeze(0)
    
    # Create sphere features: (1, num_spheres, 4)
    sphere_features = create_dummy_sphere_features(num_spheres)
    sphere_tensor = torch.tensor(sphere_features).unsqueeze(0)
    
    # Forward pass
    with torch.no_grad():
        forces = model(motion_tensor, sphere_tensor)
    
    print(f"\nModel Summary:")
    print(f"  Input: {seq_len} frames × {input_dim} features")
    print(f"  Output: {forces.shape[1]} sphere forces")
    print(f"  Force range: [{forces.min():.2f}, {forces.max():.2f}]")
    print(f"  Total force: {forces.sum():.2f} N")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
