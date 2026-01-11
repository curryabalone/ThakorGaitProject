"""
Neural Contact Force Estimation Module

Option A: Temporal Encoder (LSTM) + Spatial Decoder (PointNet-style)
for learning biomechanically accurate contact forces from motion capture data.
"""

from .model import ContactForceModel
from .data import MotionDataset, load_mot_file
from .losses import ContactForceLoss

__all__ = [
    "ContactForceModel",
    "MotionDataset", 
    "load_mot_file",
    "ContactForceLoss",
]
