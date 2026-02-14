"""
TidyBot GraspGen Service — Python Client SDK

Usage:
    from client import GraspGenClient

    client = GraspGenClient("http://<backend-host>:8002")

    # Check service health
    health = client.health()
    print(health)

    # Generate grasps from a depth image file (16-bit PNG, values in mm)
    grasps = client.generate("depth.png")
    for g in grasps:
        print(f"Score: {g['score']:.3f}, Transform:\\n{g['transform']}")

    # Generate with a target object mask
    grasps = client.generate("depth.png", mask="mask.png", num_grasps=5)

    # Full response with metadata
    result = client.generate_full("depth.png", num_grasps=20)
    print(f"Generated {result['num_grasps']} grasps in {result['inference_ms']:.0f}ms")

    # Generate from numpy depth array (float32, meters)
    import numpy as np
    depth = np.load("depth.npy")
    grasps = client.generate(depth, depth_scale=1.0)
"""

import base64
import io
import requests
import numpy as np
from pathlib import Path
from typing import Optional, Union


class GraspGenClient:
    """Client SDK for the TidyBot GraspGen 6-DOF Grasp Pose Generation Service."""

    def __init__(self, base_url: str = "http://localhost:8002", timeout: float = 60.0):
        """
        Args:
            base_url: The URL where the GraspGen service is hosted.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        """
        Check service health and GPU status.

        Returns:
            dict with keys: status, device, gpu_name, gpu_memory_mb, model_loaded, model_name
        """
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _encode_image(self, image) -> str:
        """Encode image to base64 from file path, bytes, numpy array, or pass through if already base64."""
        if isinstance(image, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, image)
            return base64.b64encode(buf.getvalue()).decode()
        elif isinstance(image, (str, Path)):
            return base64.b64encode(Path(image).read_bytes()).decode()
        elif isinstance(image, bytes):
            return base64.b64encode(image).decode()
        return image  # assume already base64

    def generate(
        self,
        depth_image,
        mask=None,
        camera_matrix: Optional[list[list[float]]] = None,
        num_grasps: int = 10,
        z_range: list[float] = [0.2, 1.8],
        depth_scale: float = 0.001,
    ) -> list[dict]:
        """
        Generate 6-DOF grasp poses from a depth image. Returns grasps only.

        Args:
            depth_image: Depth image as file path (str/Path), raw bytes, numpy array, or base64 string.
                         For file paths: expects 16-bit PNG (values in mm by default).
                         For numpy arrays: expects float32 (set depth_scale=1.0 if already in meters).
            mask: Optional binary mask as file path, bytes, numpy array, or base64 string.
                  White (255) = target object, black (0) = background.
            camera_matrix: 3x3 camera intrinsic matrix. Defaults to RealSense D435 (640x480).
            num_grasps: Maximum number of grasps to return (1-200).
            z_range: [min, max] depth range in meters to filter point cloud.
            depth_scale: Scale factor to convert depth values to meters.

        Returns:
            List of grasp dicts, each with:
                - transform: 4x4 homogeneous transform (list of lists)
                - score: float quality score (higher = better)
                - contact_point: [x, y, z] in camera frame (meters)
                - gripper_opening: float gripper width (meters)
        """
        result = self.generate_full(
            depth_image, mask=mask, camera_matrix=camera_matrix,
            num_grasps=num_grasps, z_range=z_range, depth_scale=depth_scale,
        )
        return result["grasps"]

    def generate_full(
        self,
        depth_image,
        mask=None,
        camera_matrix: Optional[list[list[float]]] = None,
        num_grasps: int = 10,
        z_range: list[float] = [0.2, 1.8],
        depth_scale: float = 0.001,
    ) -> dict:
        """
        Generate grasps with full metadata.

        Args:
            Same as generate().

        Returns:
            dict with keys:
                - grasps: list of grasp poses
                - num_grasps: int
                - device: str (cuda/cpu)
                - inference_ms: float
                - point_cloud_size: int
        """
        payload = {
            "depth_image": self._encode_image(depth_image),
            "num_grasps": num_grasps,
            "z_range": z_range,
            "depth_scale": depth_scale,
        }
        if mask is not None:
            payload["mask"] = self._encode_image(mask)
        if camera_matrix is not None:
            payload["camera_matrix"] = camera_matrix

        r = requests.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_transforms_as_numpy(self, grasps: list[dict]) -> np.ndarray:
        """
        Convert list of grasp dicts to Nx4x4 numpy array of transforms.

        Args:
            grasps: List of grasp dicts from generate().

        Returns:
            np.ndarray of shape (N, 4, 4) — homogeneous transforms ready for Franka arm.
        """
        return np.array([g["transform"] for g in grasps])


if __name__ == "__main__":
    client = GraspGenClient()
    print("Health:", client.health())
