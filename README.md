# TidyBot GraspGen Service

Backend 6-DOF grasp pose generation service for TidyBot frontend agents. Uses [Contact-GraspNet (PyTorch)](https://github.com/elchun/contact_graspnet_pytorch) to predict ranked grasp poses from depth images, running on GPU (RTX 5090).

## Service URL

```
http://158.130.109.188:8001
```

## Quick Start (Client)

**Only dependency:** `pip install requests numpy`

```python
from client import GraspGenClient

client = GraspGenClient("http://158.130.109.188:8001")

# Generate grasps from a 16-bit depth PNG (values in mm)
grasps = client.generate("depth.png")
for g in grasps:
    print(f"Score: {g['score']:.3f}")
    print(f"Transform:\n{g['transform']}")

# Generate with a target object mask (focus grasps on one object)
grasps = client.generate("depth.png", mask="object_mask.png", num_grasps=5)

# Generate from a numpy depth array (float32, meters)
import numpy as np
depth = np.load("depth.npy")  # float32, values in meters
grasps = client.generate(depth, depth_scale=1.0)

# Get transforms as Nx4x4 numpy array (ready for Franka arm)
transforms = client.get_transforms_as_numpy(grasps)
print(transforms.shape)  # (N, 4, 4)

# Full response with metadata
result = client.generate_full("depth.png", num_grasps=20)
print(f"Generated {result['num_grasps']} grasps in {result['inference_ms']:.0f}ms")

# Check service health
print(client.health())
```

## API Reference

### `GET /health`

Returns service status, GPU info, and model status.

**Response:**
```json
{
  "status": "ok",
  "device": "cuda",
  "gpu_name": "NVIDIA GeForce RTX 5090",
  "gpu_memory_mb": 32084,
  "model_loaded": true,
  "model_name": "Contact-GraspNet (PyTorch)"
}
```

### `POST /generate`

Generate ranked 6-DOF grasp poses from a depth image.

**Request:**
```json
{
  "depth_image": "<base64-encoded-depth>",
  "mask": "<base64-encoded-mask-or-null>",
  "camera_matrix": [[616.36, 0, 310.26], [0, 616.20, 236.60], [0, 0, 1]],
  "num_grasps": 10,
  "z_range": [0.2, 1.8],
  "depth_scale": 0.001
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `depth_image` | string | **required** | Base64-encoded 16-bit PNG depth image (mm) or float32 `.npy` (meters) |
| `mask` | string | `null` | Base64-encoded binary mask PNG (255=object, 0=background) |
| `camera_matrix` | float[][] | RealSense D435 | 3x3 camera intrinsic matrix `[[fx,0,cx],[0,fy,cy],[0,0,1]]` |
| `num_grasps` | int | `10` | Max number of grasps to return (1-200) |
| `z_range` | float[] | `[0.2, 1.8]` | Min/max depth in meters for point cloud filtering |
| `depth_scale` | float | `0.001` | Scale to convert depth values to meters (0.001 for mm) |

**Response:**
```json
{
  "grasps": [
    {
      "transform": [
        [r00, r01, r02, tx],
        [r10, r11, r12, ty],
        [r20, r21, r22, tz],
        [0, 0, 0, 1]
      ],
      "score": 0.95,
      "contact_point": [0.12, -0.05, 0.73],
      "gripper_opening": 0.04
    }
  ],
  "num_grasps": 10,
  "device": "cuda",
  "inference_ms": 145.23,
  "point_cloud_size": 20000
}
```

### Grasp Pose Fields

| Field | Type | Description |
|-------|------|-------------|
| `transform` | float[4][4] | 4x4 homogeneous transform — grasp pose in camera frame. Directly usable with Franka arm (after camera-to-base transform). |
| `score` | float | Grasp quality score (0-1, higher = more stable grasp) |
| `contact_point` | float[3] | 3D contact point [x, y, z] in camera frame (meters) |
| `gripper_opening` | float | Predicted gripper opening width (meters) |

### Transform Convention

The 4x4 transform follows standard robotics convention:
- **Rotation** (3x3 upper-left): Grasp orientation in camera frame
- **Translation** (3x1 right column): Grasp position in camera frame (meters)
- **Z-axis**: Approach direction (gripper closes along this axis)
- To use with Franka: multiply by your camera-to-base-frame transform (`T_base_grasp = T_base_cam @ T_cam_grasp`)

### Depth Image Formats

| Format | Encoding | Values | depth_scale |
|--------|----------|--------|-------------|
| 16-bit PNG | Base64 | Millimeters (uint16) | `0.001` (default) |
| Float32 .npy | Base64 | Meters (float32) | `1.0` |

## Server Setup

```bash
# Clone and setup
git clone https://github.com/TidyBot-Services/graspgen-service.git
cd graspgen-service
bash setup.sh

# Run
python main.py
# Or: uvicorn main:app --host 0.0.0.0 --port 8001
```

### Interactive Docs

Visit `http://158.130.109.188:8001/docs` for auto-generated Swagger UI.

## Model

**Contact-GraspNet** (Sundermeyer et al., ICRA 2021) — Efficient 6-DoF grasp generation in cluttered scenes. Predicts a full grasp distribution from a single-view depth/point cloud.

- [Paper](https://arxiv.org/abs/2103.14127)
- [Original TF code (NVIDIA)](https://github.com/NVlabs/contact_graspnet)
- [PyTorch port](https://github.com/elchun/contact_graspnet_pytorch)

## Discovery

This service is registered in `catalog.json` at:
https://github.com/TidyBot-Services/backend_wishlist/blob/main/catalog.json
