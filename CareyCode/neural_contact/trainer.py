"""
Training loop for neural contact force model.
"""

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from typing import Dict, Optional, Callable, Tuple
import numpy as np
from pathlib import Path
import pickle
from dataclasses import dataclass

from .model import ContactForceModel, create_model
from .losses import ContactForceLoss, create_training_loss
from .data import MotionDataset


@dataclass
class TrainConfig:
    """Training configuration."""
    # Model
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    decoder_hidden_dims: Tuple[int, ...] = (128, 64)
    history_length: int = 10
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    warmup_steps: int = 100
    
    # Loss weights
    kinematic_weight: float = 1.0
    smoothness_weight: float = 0.1
    physics_weight: float = 0.1
    sparsity_weight: float = 0.01
    
    # Subject
    body_mass: float = 70.0
    
    # Checkpointing
    checkpoint_dir: Optional[str] = None
    save_every: int = 10


class TrainState(train_state.TrainState):
    """Extended train state with additional fields."""
    pass


def create_train_state(
    model: ContactForceModel,
    params: dict,
    learning_rate: float,
    warmup_steps: int = 100,
) -> TrainState:
    """Create optimizer and training state."""
    
    # Learning rate schedule with warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=10000,
        end_value=learning_rate * 0.01,
    )
    
    # Optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=1e-4),
    )
    
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


@jax.jit
def train_step(
    state: TrainState,
    pose_batch: jnp.ndarray,
    sphere_positions: jnp.ndarray,
    sphere_radii: jnp.ndarray,
    gait_phase: jnp.ndarray,
    loss_fn: Callable,
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """
    Single training step.
    
    Args:
        state: current training state
        pose_batch: (batch, seq_len, pose_dim) pose history
        sphere_positions: (num_spheres, 3)
        sphere_radii: (num_spheres,)
        gait_phase: (batch,) gait phase values
        loss_fn: loss function
        
    Returns:
        updated state, loss dict
    """
    def compute_loss(params):
        # Forward pass
        forces = state.apply_fn(
            params,
            pose_batch,
            sphere_positions,
            sphere_radii,
            training=True,
        )
        
        # Compute losses (without simulation for now)
        # measured_motion is the last frame's pose
        measured_motion = pose_batch[:, -1, :]
        
        losses = loss_fn(
            predicted_forces=forces,
            measured_motion=measured_motion,
            simulated_motion=None,  # Will add MJX simulation later
            gait_phase=gait_phase,
        )
        
        return losses['total'], losses
    
    # Compute gradients
    (loss, losses), grads = jax.value_and_grad(compute_loss, has_aux=True)(state.params)
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    
    return state, losses



class Trainer:
    """Training manager for contact force model."""
    
    def __init__(
        self,
        config: TrainConfig,
        dataset: MotionDataset,
        sphere_positions: np.ndarray,
        sphere_radii: np.ndarray,
    ):
        self.config = config
        self.dataset = dataset
        self.sphere_positions = jnp.array(sphere_positions)
        self.sphere_radii = jnp.array(sphere_radii)
        
        # Create model
        self.model, params = create_model(
            pose_dim=dataset.pose_dim,
            num_spheres=len(sphere_radii),
            lstm_hidden_size=config.lstm_hidden_size,
            lstm_num_layers=config.lstm_num_layers,
            decoder_hidden_dims=config.decoder_hidden_dims,
        )
        
        # Create training state
        self.state = create_train_state(
            self.model,
            params,
            config.learning_rate,
            config.warmup_steps,
        )
        
        # Create loss function
        self.loss_fn = create_training_loss(
            body_mass=config.body_mass,
            kinematic_weight=config.kinematic_weight,
            smoothness_weight=config.smoothness_weight,
            physics_weight=config.physics_weight,
            sparsity_weight=config.sparsity_weight,
        )
        
        # RNG
        self.rng = np.random.default_rng(42)
        
        # History
        self.history = {'train_loss': [], 'components': []}
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        n_batches = len(self.dataset) // self.config.batch_size
        epoch_losses = []
        
        for _ in range(n_batches):
            # Get batch
            pose_batch = self.dataset.random_batch(
                self.config.batch_size, 
                self.rng
            )
            
            # Dummy gait phase for now (will compute from data)
            gait_phase = jnp.linspace(0, 1, self.config.batch_size)
            
            # Training step
            self.state, losses = train_step(
                self.state,
                pose_batch,
                self.sphere_positions,
                self.sphere_radii,
                gait_phase,
                self.loss_fn,
            )
            
            epoch_losses.append({k: float(v) for k, v in losses.items()})
        
        # Average losses
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([l[key] for l in epoch_losses])
        
        return avg_losses
    
    def train(self, verbose: bool = True) -> Dict[str, list]:
        """Full training loop."""
        
        for epoch in range(self.config.num_epochs):
            losses = self.train_epoch()
            
            self.history['train_loss'].append(losses['total'])
            self.history['components'].append(losses)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                print(f"  Total: {losses['total']:.4f}")
                print(f"  Smoothness: {losses['smoothness']:.4f}")
                print(f"  Physics: {losses['physics']:.4f}")
                print(f"  Sparsity: {losses['sparsity']:.4f}")
            
            # Checkpoint
            if self.config.checkpoint_dir and (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch + 1)
        
        return self.history
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        if self.config.checkpoint_dir is None:
            return
        
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'params': self.state.params,
            'config': self.config,
            'history': self.history,
        }
        
        path = checkpoint_dir / f"checkpoint_{epoch:04d}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.state = self.state.replace(params=checkpoint['params'])
        self.history = checkpoint['history']
        
        return checkpoint['epoch']
    
    def predict(self, pose_sequence: np.ndarray) -> np.ndarray:
        """
        Predict forces for a pose sequence.
        
        Args:
            pose_sequence: (seq_len, pose_dim) full motion sequence
            
        Returns:
            forces: (seq_len, num_spheres) predicted forces
        """
        forces = self.model.apply(
            self.state.params,
            jnp.array(pose_sequence),
            self.sphere_positions,
            self.sphere_radii,
            method=self.model.predict_sequence,
            history_length=self.config.history_length,
        )
        return np.array(forces)
