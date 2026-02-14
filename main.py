"""
GraspGen Service — TidyBot Backend
6-DOF Grasp Pose Generation using Contact-GraspNet (PyTorch).
Hosted on FastAPI. Accepts depth images and returns ranked grasp poses as 4x4 transforms.
"""

import base64
import io
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

# ─── Paths ────────────────────────────────────────────────────────
CGN_DIR = os.path.join(os.path.dirname(__file__), "contact_graspnet_pytorch")
CKPT_DIR = os.path.join(CGN_DIR, "checkpoints", "contact_graspnet")

# Add CGN and dependencies to path for imports
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
POINTNET_DIR = os.path.join(SERVICE_DIR, "Pointnet_Pointnet2_pytorch")
for p in [SERVICE_DIR, POINTNET_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Globals (loaded at startup) ─────────────────────────────────
grasp_estimator = None
global_config = None

# Default RealSense D435 intrinsics (640x480) — clients can override
DEFAULT_K = np.array([
    [616.36529541, 0.0, 310.25881958],
    [0.0, 616.20294189, 236.59980774],
    [0.0, 0.0, 1.0],
])


def load_model():
    """Load Contact-GraspNet model and weights."""
    global grasp_estimator, global_config
    from contact_graspnet_pytorch import config_utils
    from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
    from contact_graspnet_pytorch.checkpoints import CheckpointIO

    global_config = config_utils.load_config(CKPT_DIR, batch_size=1, arg_configs=[])
    grasp_estimator = GraspEstimator(global_config)

    model_checkpoint_dir = os.path.join(CKPT_DIR, "checkpoints")
    checkpoint_io = CheckpointIO(checkpoint_dir=model_checkpoint_dir, model=grasp_estimator.model)
    try:
        checkpoint_io.load("model.pt")
    except FileExistsError:
        print("WARNING: No model checkpoint found. Model will produce random outputs.")

    grasp_estimator.model.eval()
    print(f"Contact-GraspNet loaded on {DEVICE}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading Contact-GraspNet on {DEVICE}...")
    load_model()
    print("Ready.")
    yield


# ─── FastAPI App ──────────────────────────────────────────────────
app = FastAPI(
    title="TidyBot GraspGen Service",
    description="6-DOF grasp pose generation service for TidyBot. "
                "Uses Contact-GraspNet to predict ranked grasp poses from depth images.",
    version="0.1.0",
    lifespan=lifespan,
)


# ─── Schemas ──────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    depth_image: str = Field(..., description="Base64-encoded 16-bit PNG depth image (values in mm) or float32 .npy (values in meters)")
    mask: Optional[str] = Field(None, description="Base64-encoded binary mask PNG (255=object, 0=background). Used to filter grasps to target object.")
    camera_matrix: Optional[list[list[float]]] = Field(
        None,
        description="3x3 camera intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]. Defaults to RealSense D435 (640x480).",
    )
    num_grasps: int = Field(10, ge=1, le=200, description="Maximum number of grasp poses to return")
    z_range: list[float] = Field([0.2, 1.8], description="Min/max depth range in meters to filter point cloud")
    depth_scale: float = Field(0.001, description="Scale factor to convert depth values to meters (0.001 for mm, 1.0 for meters)")


class GraspPose(BaseModel):
    transform: list[list[float]] = Field(..., description="4x4 homogeneous transform matrix (grasp pose in camera frame)")
    score: float = Field(..., description="Grasp quality score (higher is better)")
    contact_point: list[float] = Field(..., description="3D contact point [x, y, z] in camera frame (meters)")
    gripper_opening: float = Field(..., description="Predicted gripper opening width in meters")


class GenerateResponse(BaseModel):
    grasps: list[GraspPose]
    num_grasps: int
    device: str
    inference_ms: float
    point_cloud_size: int


class HealthResponse(BaseModel):
    status: str
    device: str
    gpu_name: Optional[str]
    gpu_memory_mb: Optional[int]
    model_loaded: bool
    model_name: str


# ─── Endpoints ────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    """Check service health and GPU status."""
    gpu_name = None
    gpu_mem = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = int(torch.cuda.get_device_properties(0).total_mem / 1024 / 1024) if hasattr(torch.cuda.get_device_properties(0), 'total_mem') else int(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
    return HealthResponse(
        status="ok",
        device=DEVICE,
        gpu_name=gpu_name,
        gpu_memory_mb=gpu_mem,
        model_loaded=grasp_estimator is not None,
        model_name="Contact-GraspNet (PyTorch)",
    )


def decode_depth(b64_data: str, depth_scale: float) -> np.ndarray:
    """Decode base64 depth image to float32 array in meters."""
    raw = base64.b64decode(b64_data)
    
    # Try as .npy first
    try:
        arr = np.load(io.BytesIO(raw))
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            return arr.astype(np.float32)
        return arr.astype(np.float32) * depth_scale
    except Exception:
        pass
    
    # Try as image (16-bit PNG or similar)
    try:
        img = np.array(Image.open(io.BytesIO(raw)))
        return img.astype(np.float32) * depth_scale
    except Exception as e:
        raise ValueError(f"Cannot decode depth image: {e}")


def decode_mask(b64_data: str) -> np.ndarray:
    """Decode base64 mask to binary array."""
    raw = base64.b64decode(b64_data)
    img = np.array(Image.open(io.BytesIO(raw)))
    if len(img.shape) == 3:
        img = img[:, :, 0]
    return (img > 127).astype(np.uint8)


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate 6-DOF grasp poses from a depth image."""
    if grasp_estimator is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Decode depth
    try:
        depth = decode_depth(request.depth_image, request.depth_scale)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid depth image: {e}")

    # Camera matrix
    K = np.array(request.camera_matrix) if request.camera_matrix else DEFAULT_K

    # Decode optional mask → segmap
    segmap = None
    if request.mask:
        try:
            mask = decode_mask(request.mask)
            # Convert binary mask to labeled segmap (1 = target object, 0 = background)
            segmap = mask.astype(np.int32)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid mask: {e}")

    t0 = time.perf_counter()

    # Extract point cloud from depth
    z_range = request.z_range
    pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(
        depth, K, segmap=segmap, z_range=z_range, skip_border_objects=False,
    )

    pc_size = pc_full.shape[0] if pc_full is not None else 0

    if pc_full is None or pc_size == 0:
        raise HTTPException(status_code=400, detail="No valid points in depth image after filtering")

    # Generate grasps
    local_regions = segmap is not None and len(pc_segments) > 0
    filter_grasps = segmap is not None and len(pc_segments) > 0

    pred_grasps_cam, scores, contact_pts, gripper_openings = grasp_estimator.predict_scene_grasps(
        pc_full,
        pc_segments=pc_segments if local_regions else {},
        local_regions=local_regions,
        filter_grasps=filter_grasps,
        forward_passes=1,
    )

    inference_ms = (time.perf_counter() - t0) * 1000

    # Collect all grasps across segments, sort by score
    all_grasps = []
    for seg_id in pred_grasps_cam:
        grasps_arr = pred_grasps_cam[seg_id]
        scores_arr = scores[seg_id]
        contacts_arr = contact_pts[seg_id]
        openings_arr = gripper_openings[seg_id]

        if len(grasps_arr) == 0:
            continue

        for i in range(len(grasps_arr)):
            contact = contacts_arr[i] if len(contacts_arr.shape) > 1 else contacts_arr
            opening = float(openings_arr[i]) if hasattr(openings_arr, '__len__') and len(openings_arr.shape) > 0 else float(openings_arr)
            all_grasps.append({
                "transform": grasps_arr[i],
                "score": float(scores_arr[i]),
                "contact_point": contact.tolist() if hasattr(contact, 'tolist') else list(contact),
                "gripper_opening": opening,
            })

    # Sort by score descending, take top N
    all_grasps.sort(key=lambda g: g["score"], reverse=True)
    all_grasps = all_grasps[: request.num_grasps]

    # Convert numpy arrays to lists for JSON serialization
    result_grasps = []
    for g in all_grasps:
        transform = g["transform"]
        if hasattr(transform, "tolist"):
            transform = transform.tolist()
        result_grasps.append(
            GraspPose(
                transform=transform,
                score=g["score"],
                contact_point=g["contact_point"][:3],
                gripper_opening=g["gripper_opening"],
            )
        )

    return GenerateResponse(
        grasps=result_grasps,
        num_grasps=len(result_grasps),
        device=DEVICE,
        inference_ms=round(inference_ms, 2),
        point_cloud_size=pc_size,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
