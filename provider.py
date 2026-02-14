"""Minimal provider stub for Contact-GraspNet inference (training augmentation only)."""
import numpy as np

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """Add random jitter to point cloud (training augmentation)."""
    N, C = batch_data.shape[1], batch_data.shape[2]
    jitter = np.clip(sigma * np.random.randn(*batch_data.shape), -clip, clip)
    return batch_data + jitter
