"""实验八的集中配置。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

WORK8_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = WORK8_DIR / "models"
DEFAULT_OUTPUT_DIR = WORK8_DIR / "outputs"

MODEL_TYPE = "smpl"
MODEL_GENDER = "neutral"
MODEL_EXTENSION = "pkl"
DEFAULT_NUM_BETAS = 10
DEFAULT_WEIGHT_JOINT_ID = 18

# SMPL 的 24 个骨骼关节编号。body_pose 不包含 0 号根关节。
SMPL_JOINT_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)

# 形状参数：只修改前 3 个 beta，其余保持为 0。
DEMO_BETA_VALUES: tuple[float, ...] = (2.0, -1.2, 0.8)

# 静态展示姿态。键为 SMPL 关节编号，值为轴角向量，单位为弧度。
DEMO_BODY_POSE: dict[int, tuple[float, float, float]] = {
    1: (0.25, 0.0, 0.08),
    2: (-0.18, 0.0, -0.08),
    4: (0.35, 0.0, 0.0),
    5: (0.20, 0.0, 0.0),
    16: (0.0, 0.0, 0.45),
    17: (0.0, 0.0, -0.45),
    18: (0.0, -0.35, 0.0),
    19: (0.0, 0.35, 0.0),
}

# 选做动画默认参数：固定体型，驱动左肘绕 y 轴弯曲。
DEFAULT_ANIMATION_JOINT_ID = 18
DEFAULT_ANIMATION_AXIS = "y"
DEFAULT_ANIMATION_ANGLE_DEGREES = -75.0
DEFAULT_ANIMATION_FORWARD_FRAMES = 36
DEFAULT_ANIMATION_FPS = 24
DEFAULT_ANIMATION_FORMAT = "both"
DEFAULT_ANIMATION_PING_PONG = True
DEFAULT_KEEP_ANIMATION_FRAMES = True

# Matplotlib 渲染参数。
FIGURE_DPI = 220
ANIMATION_DPI = 150
CAMERA_ELEVATION = 12.0
CAMERA_AZIMUTH = 108.0
CAMERA_FOCAL_LENGTH = 0.85
DEFAULT_MESH_RGBA = np.array([0.82, 0.67, 0.52, 1.0], dtype=np.float64)
DEFAULT_COLORMAP = "viridis"
POSE_OFFSET_COLORMAP = "magma"
