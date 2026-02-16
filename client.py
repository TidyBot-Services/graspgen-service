"""
TidyBot GraspGen Service — Python Client SDK

Usage:
    from service_clients.graspgen.client import GraspGenClient

    client = GraspGenClient()
    grasps = client.generate(depth_bytes, num_grasps=10)
    for g in grasps:
        print(f"Score: {g['score']:.3f}")
"""

import base64
import io
import json
import urllib.request
import urllib.error
import numpy as np
from pathlib import Path
from typing import Optional


class GraspGenClient:
    """Client SDK for the TidyBot GraspGen 6-DOF Grasp Pose Generation Service."""

    def __init__(self, host: str = "http://158.130.109.188:8002", timeout: float = 60.0):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.host}{path}", data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(f"{self.host}{path}")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def health(self) -> dict:
        """Check service health and GPU status."""
        return self._get("/health")

    @staticmethod
    def _encode_image(image) -> str:
        """Encode image to base64 from file path, bytes, numpy array, or pass through if already base64."""
        if isinstance(image, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, image)
            return base64.b64encode(buf.getvalue()).decode()
        elif isinstance(image, (str, Path)):
            p = Path(image)
            if p.exists():
                return base64.b64encode(p.read_bytes()).decode()
            return image
        elif isinstance(image, bytes):
            return base64.b64encode(image).decode()
        return image

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
            depth_image: Depth image as file path, raw bytes, numpy array, or base64 string.
            mask: Optional binary mask (same formats).
            camera_matrix: 3x3 camera intrinsic matrix.
            num_grasps: Maximum number of grasps to return (1-200).
            z_range: [min, max] depth range in meters.
            depth_scale: Scale factor to convert depth values to meters.

        Returns:
            List of dicts with keys: transform (4x4), score, contact_point [x,y,z], gripper_opening.
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

        Returns:
            dict with keys: grasps, num_grasps, device, inference_ms, point_cloud_size.
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
        return self._post("/generate", payload)

    @staticmethod
    def get_transforms_as_numpy(grasps: list[dict]) -> np.ndarray:
        """Convert list of grasp dicts to Nx4x4 numpy array of transforms."""
        return np.array([g["transform"] for g in grasps])


if __name__ == "__main__":
    client = GraspGenClient()
    print("Health:", client.health())
