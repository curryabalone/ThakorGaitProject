"""
Neural Contact Force Model - Option A Architecture

Temporal Encoder (LSTM) + Spatial Decoder (PointNet-style MLP)
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Tuple
import numpy as np


class TemporalEncoder(nn.Module):
    """LSTM-based encoder for capturing gait dynamics from pose history."""
    
    hidden_size: int = 256
    num_layers: int = 2
    
    @nn.compact
    def __call__(self, pose_sequence: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Args:
            pose_sequence: (batch, seq_len, pose_dim) - history of foot poses
            training: whether in training mode
            
        Returns:
            latent_state: (batch, hidden_size) - encoded gait state
        """
        batch_size = pose_sequence.shape[0]
        
        # Stack LSTM layers
        carry = None
        x = pose_sequence
        
        for i in range(self.num_layers):
            lstm = nn.OptimizedLSTMCell(features=self.hidden_size, name=f"lstm_{i}")
            
            # Initialize carry if needed
            if carry is None:
                carry = lstm.initialize_carry(jax.random.PRNGKey(0), (batch_size, self.hidden_size))
            
            # Process sequence
            outputs = []
            for t in range(x.shape[1]):
                carry, y = lstm(carry, x[:, t, :] if i == 0 else x[:, t, :])
                outputs.append(y)
            
            x = jnp.stack(outputs, axis=1)
            
            if i < self.num_layers - 1:
                x = nn.Dropout(rate=0.1, deterministic=not training)(x)
        
        # Return final hidden state
        return x[:, -1, :]


class SpatialDecoder(nn.Module):
    """PointNet-style MLP decoder for per-sphere force prediction."""
    
    hidden_dims: Sequence[int] = (128, 64)
    
    @nn.compact
    def __call__(self, 
                 latent_state: jnp.ndarray,
                 sphere_positions: jnp.ndarray,
                 sphere_radii: jnp.ndarray,
                 training: bool = True) -> jnp.ndarray:
        """
        Args:
            latent_state: (batch, latent_dim) - encoded gait state
            sphere_positions: (num_spheres, 3) - sphere center positions
            sphere_radii: (num_spheres,) - sphere radii
            training: whether in training mode
            
        Returns:
            forces: (batch, num_spheres) - predicted force magnitude per sphere
        """
        batch_size = latent_state.shape[0]
        num_spheres = sphere_positions.shape[0]
        
        # Combine sphere features: position (3) + radius (1) = 4
        sphere_features = jnp.concatenate([
            sphere_positions,
            sphere_radii[:, None]
        ], axis=-1)  # (num_spheres, 4)
        
        # Broadcast latent state to all spheres
        latent_expanded = jnp.tile(latent_state[:, None, :], (1, num_spheres, 1))
        # (batch, num_spheres, latent_dim)
        
        # Broadcast sphere features to all batches
        sphere_expanded = jnp.tile(sphere_features[None, :, :], (batch_size, 1, 1))
        # (batch, num_spheres, 4)
        
        # Concatenate: (batch, num_spheres, latent_dim + 4)
        x = jnp.concatenate([sphere_expanded, latent_expanded], axis=-1)
        
        # Reshape for MLP: (batch * num_spheres, features)
        x = x.reshape(-1, x.shape[-1])
        
        # MLP layers
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f"dense_{i}")(x)
            x = nn.relu(x)
            if training:
                x = nn.Dropout(rate=0.1, deterministic=not training)(x)
        
        # Output layer - single force magnitude per sphere
        x = nn.Dense(1, name="output")(x)
        
        # Softplus ensures non-negative forces
        x = nn.softplus(x)
        
        # Reshape back: (batch, num_spheres)
        forces = x.reshape(batch_size, num_spheres)
        
        return forces



class ContactForceModel(nn.Module):
    """
    Full contact force prediction model.
    
    Architecture (Option A):
        foot_pose_history -> LSTM Encoder -> latent_state
        latent_state + sphere_geometry -> Spatial MLP -> per_sphere_forces
    """
    
    # Temporal encoder params
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    
    # Spatial decoder params
    decoder_hidden_dims: Sequence[int] = (128, 64)
    
    def setup(self):
        self.temporal_encoder = TemporalEncoder(
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
        )
        self.spatial_decoder = SpatialDecoder(
            hidden_dims=self.decoder_hidden_dims,
        )
    
    def __call__(self,
                 pose_history: jnp.ndarray,
                 sphere_positions: jnp.ndarray,
                 sphere_radii: jnp.ndarray,
                 training: bool = True) -> jnp.ndarray:
        """
        Forward pass: predict contact forces from pose history and sphere geometry.
        
        Args:
            pose_history: (batch, seq_len, pose_dim) - foot pose history
                pose_dim typically includes:
                - position (3)
                - orientation quaternion (4) 
                - joint angles (n)
            sphere_positions: (num_spheres, 3) - contact sphere positions in foot frame
            sphere_radii: (num_spheres,) - contact sphere radii
            training: whether in training mode
            
        Returns:
            forces: (batch, num_spheres) - predicted normal force magnitude per sphere
        """
        # Encode temporal dynamics
        latent_state = self.temporal_encoder(pose_history, training=training)
        
        # Decode to per-sphere forces
        forces = self.spatial_decoder(
            latent_state, 
            sphere_positions, 
            sphere_radii,
            training=training
        )
        
        return forces
    
    def predict_sequence(self,
                         full_pose_sequence: jnp.ndarray,
                         sphere_positions: jnp.ndarray,
                         sphere_radii: jnp.ndarray,
                         history_length: int = 10) -> jnp.ndarray:
        """
        Predict forces for an entire motion sequence using sliding window.
        
        Args:
            full_pose_sequence: (seq_len, pose_dim) - full motion sequence
            sphere_positions: (num_spheres, 3)
            sphere_radii: (num_spheres,)
            history_length: number of frames of history to use
            
        Returns:
            forces: (seq_len, num_spheres) - forces for each frame
        """
        seq_len = full_pose_sequence.shape[0]
        all_forces = []
        
        for t in range(seq_len):
            # Get history window (pad with first frame if needed)
            start_idx = max(0, t - history_length + 1)
            history = full_pose_sequence[start_idx:t+1]
            
            # Pad if needed
            if history.shape[0] < history_length:
                pad_size = history_length - history.shape[0]
                padding = jnp.tile(history[0:1], (pad_size, 1))
                history = jnp.concatenate([padding, history], axis=0)
            
            # Add batch dimension and predict
            history_batch = history[None, :, :]  # (1, history_length, pose_dim)
            forces = self(history_batch, sphere_positions, sphere_radii, training=False)
            all_forces.append(forces[0])  # Remove batch dim
        
        return jnp.stack(all_forces, axis=0)


def create_model(
    pose_dim: int,
    num_spheres: int,
    lstm_hidden_size: int = 256,
    lstm_num_layers: int = 2,
    decoder_hidden_dims: Sequence[int] = (128, 64),
    seed: int = 42,
) -> Tuple[ContactForceModel, dict]:
    """
    Create and initialize the contact force model.
    
    Args:
        pose_dim: dimension of pose vector
        num_spheres: number of contact spheres
        lstm_hidden_size: LSTM hidden state size
        lstm_num_layers: number of LSTM layers
        decoder_hidden_dims: hidden layer sizes for spatial decoder
        seed: random seed for initialization
        
    Returns:
        model: ContactForceModel instance
        params: initialized parameters
    """
    model = ContactForceModel(
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        decoder_hidden_dims=decoder_hidden_dims,
    )
    
    # Create dummy inputs for initialization
    key = jax.random.PRNGKey(seed)
    dummy_pose_history = jnp.zeros((1, 10, pose_dim))  # (batch, seq, pose_dim)
    dummy_sphere_pos = jnp.zeros((num_spheres, 3))
    dummy_sphere_radii = jnp.ones((num_spheres,)) * 0.01
    
    # Initialize parameters
    params = model.init(
        key,
        dummy_pose_history,
        dummy_sphere_pos,
        dummy_sphere_radii,
        training=False,
    )
    
    return model, params
