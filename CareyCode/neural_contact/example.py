#!/usr/bin/env python3
"""
Example: Train neural contact force model on gait data.

Usage:
    cd CareyCode
    python -m neural_contact.example --mot ../Session1/OpenSimData/Kinematics/0_8_m_s.mot
"""

import argparse
import numpy as np
from pathlib import Path

from .data import MotionDataset, load_sphere_geometry
from .trainer import Trainer, TrainConfig


def create_dummy_spheres(num_spheres: int = 50) -> tuple:
    """Create dummy contact sphere geometry for testing."""
    # Grid of spheres on foot sole (simplified)
    # Real implementation would load from MuJoCo model
    
    x = np.linspace(-0.05, 0.15, 10)  # heel to toe
    y = np.linspace(-0.03, 0.03, 5)   # medial to lateral
    
    positions = []
    for xi in x:
        for yi in y:
            positions.append([xi, yi, -0.02])  # below foot
    
    positions = np.array(positions)[:num_spheres]
    radii = np.ones(len(positions)) * 0.008  # 8mm radius
    
    return positions, radii


def main():
    parser = argparse.ArgumentParser(description='Train contact force model')
    parser.add_argument('--mot', type=str, nargs='+',
                        default=['../Session1/OpenSimData/Kinematics/0_8_m_s.mot'],
                        help='Path(s) to .mot motion file(s)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to MuJoCo XML (for sphere geometry)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--output', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Neural Contact Force Estimation - Training")
    print("=" * 60)
    
    # Load motion data
    print(f"\nLoading motion data from {len(args.mot)} file(s)...")
    dataset = MotionDataset(
        mot_files=args.mot,
        history_length=10,
        stride=1,
        foot='right',
    )
    print(f"  Sequences: {len(dataset)}")
    print(f"  Pose dimension: {dataset.pose_dim}")
    print(f"  Joints: {dataset.joint_names}")
    
    # Load or create sphere geometry
    if args.model:
        print(f"\nLoading sphere geometry from {args.model}...")
        sphere_positions, sphere_radii = load_sphere_geometry(args.model)
    else:
        print("\nUsing dummy sphere geometry (50 spheres)...")
        sphere_positions, sphere_radii = create_dummy_spheres(50)
    
    print(f"  Spheres: {len(sphere_radii)}")
    
    # Training config
    config = TrainConfig(
        lstm_hidden_size=128,  # Smaller for faster iteration
        lstm_num_layers=2,
        decoder_hidden_dims=(64, 32),
        history_length=10,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        checkpoint_dir=args.output,
        save_every=10,
    )
    
    # Create trainer
    print("\nInitializing model...")
    trainer = Trainer(
        config=config,
        dataset=dataset,
        sphere_positions=sphere_positions,
        sphere_radii=sphere_radii,
    )
    
    # Train
    print("\nTraining...")
    print("-" * 40)
    history = trainer.train(verbose=True)
    
    print("-" * 40)
    print(f"\nTraining complete!")
    print(f"Final loss: {history['train_loss'][-1]:.4f}")
    
    # Save final model
    trainer.save_checkpoint(args.epochs)
    print(f"Model saved to {args.output}/")


if __name__ == '__main__':
    main()
