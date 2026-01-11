"""
Loss functions for neural contact force estimation.

Components:
1. Kinematic tracking loss (primary)
2. Temporal smoothness loss
3. Physics plausibility loss
4. Sparsity regularization
"""

import jax
import jax.numpy as jnp
from typing import Optional, Dict, NamedTuple


class LossWeights(NamedTuple):
    """Weights for each loss component."""
    kinematic: float = 1.0
    smoothness: float = 0.1
    physics: float = 0.1
    sparsity: float = 0.01


class ContactForceLoss:
    """Combined loss function for contact force prediction."""
    
    def __init__(
        self,
        weights: Optional[LossWeights] = None,
        body_mass: float = 70.0,  # kg
        gravity: float = 9.81,    # m/s^2
    ):
        """
        Args:
            weights: loss component weights
            body_mass: subject body mass for physics loss
            gravity: gravitational acceleration
        """
        self.weights = weights or LossWeights()
        self.body_weight = body_mass * gravity  # Expected GRF during stance
    
    def kinematic_loss(
        self,
        simulated_motion: jnp.ndarray,
        measured_motion: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        MSE between simulated and measured joint angles.
        
        Args:
            simulated_motion: (batch, n_joints) simulated joint angles
            measured_motion: (batch, n_joints) measured joint angles
            
        Returns:
            scalar loss value
        """
        return jnp.mean((simulated_motion - measured_motion) ** 2)
    
    def smoothness_loss(self, forces: jnp.ndarray) -> jnp.ndarray:
        """
        Penalize rapid changes in forces between frames.
        
        Args:
            forces: (batch, num_spheres) or (seq_len, num_spheres) predicted forces
            
        Returns:
            scalar loss value
        """
        if forces.shape[0] < 2:
            return jnp.array(0.0)
        
        force_diff = forces[1:] - forces[:-1]
        return jnp.mean(force_diff ** 2)
    
    def physics_loss(
        self,
        forces: jnp.ndarray,
        gait_phase: jnp.ndarray,
        foot: str = 'right',
    ) -> jnp.ndarray:
        """
        Penalize deviation from expected total GRF based on gait phase.
        
        During stance: total force ≈ body weight (single support) or ~0.5 BW (double support)
        During swing: total force ≈ 0
        
        Args:
            forces: (batch, num_spheres) predicted forces
            gait_phase: (batch,) gait phase 0-1 (0=heel strike, 0.6≈toe off)
            foot: 'left' or 'right'
            
        Returns:
            scalar loss value
        """
        total_force = jnp.sum(forces, axis=-1)  # (batch,)
        
        # Expected GRF profile (simplified)
        # Stance phase: 0-60% of gait cycle
        # Swing phase: 60-100% of gait cycle
        stance_mask = gait_phase < 0.6
        
        # During stance, expect ~body weight (with double-hump pattern)
        # Simplified: use body weight as target during stance
        expected_stance = self.body_weight * self._grf_profile(gait_phase)
        expected_swing = jnp.zeros_like(total_force)
        
        expected_force = jnp.where(stance_mask, expected_stance, expected_swing)
        
        return jnp.mean((total_force - expected_force) ** 2)
    
    def _grf_profile(self, phase: jnp.ndarray) -> jnp.ndarray:
        """
        Approximate vertical GRF profile during stance (normalized).
        
        Classic double-hump pattern:
        - First peak at ~15% (loading response)
        - Valley at ~30% (midstance)
        - Second peak at ~45% (terminal stance)
        - Drops to 0 at ~60% (toe off)
        """
        # Normalize to stance phase (0-0.6 -> 0-1)
        stance_phase = jnp.clip(phase / 0.6, 0, 1)
        
        # Double-hump approximation using sum of Gaussians
        peak1 = 1.1 * jnp.exp(-((stance_phase - 0.25) ** 2) / 0.02)
        peak2 = 1.1 * jnp.exp(-((stance_phase - 0.75) ** 2) / 0.02)
        valley = 0.8 * jnp.exp(-((stance_phase - 0.5) ** 2) / 0.01)
        
        profile = peak1 + peak2 - valley * 0.3
        
        # Smooth onset and offset
        onset = jnp.clip(stance_phase * 5, 0, 1)
        offset = jnp.clip((1 - stance_phase) * 5, 0, 1)
        
        return profile * onset * offset
    
    def sparsity_loss(self, forces: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
        """
        Encourage sparse activation (fewer active contact points).
        
        Uses L0.5 pseudo-norm approximation.
        
        Args:
            forces: (batch, num_spheres) predicted forces
            eps: small constant for numerical stability
            
        Returns:
            scalar loss value
        """
        return jnp.mean(jnp.sqrt(forces + eps))


    def __call__(
        self,
        predicted_forces: jnp.ndarray,
        measured_motion: jnp.ndarray,
        simulated_motion: Optional[jnp.ndarray] = None,
        gait_phase: Optional[jnp.ndarray] = None,
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute combined loss.
        
        Args:
            predicted_forces: (batch, num_spheres) or (seq_len, num_spheres)
            measured_motion: (batch, n_joints) measured joint angles
            simulated_motion: (batch, n_joints) simulated joint angles (optional)
            gait_phase: (batch,) gait phase values (optional)
            
        Returns:
            dict with 'total' loss and individual components
        """
        losses = {}
        
        # Kinematic tracking (requires simulation)
        if simulated_motion is not None:
            losses['kinematic'] = self.kinematic_loss(simulated_motion, measured_motion)
        else:
            losses['kinematic'] = jnp.array(0.0)
        
        # Temporal smoothness
        losses['smoothness'] = self.smoothness_loss(predicted_forces)
        
        # Physics plausibility
        if gait_phase is not None:
            losses['physics'] = self.physics_loss(predicted_forces, gait_phase)
        else:
            losses['physics'] = jnp.array(0.0)
        
        # Sparsity
        losses['sparsity'] = self.sparsity_loss(predicted_forces)
        
        # Weighted total
        losses['total'] = (
            self.weights.kinematic * losses['kinematic'] +
            self.weights.smoothness * losses['smoothness'] +
            self.weights.physics * losses['physics'] +
            self.weights.sparsity * losses['sparsity']
        )
        
        return losses


def create_training_loss(
    body_mass: float = 70.0,
    kinematic_weight: float = 1.0,
    smoothness_weight: float = 0.1,
    physics_weight: float = 0.1,
    sparsity_weight: float = 0.01,
) -> ContactForceLoss:
    """
    Create loss function with custom weights.
    
    Args:
        body_mass: subject body mass in kg
        kinematic_weight: weight for kinematic tracking loss
        smoothness_weight: weight for temporal smoothness loss
        physics_weight: weight for physics plausibility loss
        sparsity_weight: weight for sparsity regularization
        
    Returns:
        ContactForceLoss instance
    """
    weights = LossWeights(
        kinematic=kinematic_weight,
        smoothness=smoothness_weight,
        physics=physics_weight,
        sparsity=sparsity_weight,
    )
    return ContactForceLoss(weights=weights, body_mass=body_mass)
