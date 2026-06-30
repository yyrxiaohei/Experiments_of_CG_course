"""选做部分：固定 shape，驱动单个关节并导出 GIF/MP4。"""

from __future__ import annotations

import math
import shutil
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from .config import SMPL_JOINT_NAMES
from .lbs import compute_manual_lbs, set_body_joint_pose
from .visualization import save_animation_frame, to_numpy


@dataclass(frozen=True)
class AnimationResult:
    """动画导出结果。"""

    frame_count: int
    frame_dir: Path | None
    gif_path: Path | None
    mp4_path: Path | None


def build_angle_sequence(
    target_angle_degrees: float,
    forward_frames: int,
    ping_pong: bool,
) -> np.ndarray:
    """使用余弦缓入缓出，从 0 平滑旋转到目标角度。"""

    if forward_frames < 2:
        raise ValueError("animation_frames 至少为 2")

    progress = np.linspace(0.0, 1.0, forward_frames, dtype=np.float64)
    eased_progress = 0.5 - 0.5 * np.cos(np.pi * progress)
    forward = target_angle_degrees * eased_progress

    if not ping_pong:
        return forward

    # 反向序列不重复最大角度，最后回到 0，循环播放时不会突跳。
    return np.concatenate([forward, forward[-2::-1]])


def axis_angle_from_degrees(axis: str, angle_degrees: float) -> tuple[float, float, float]:
    """将指定坐标轴和角度转换为轴角向量。"""

    axis_to_index = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_to_index:
        raise ValueError(f"animation_axis 应为 x、y 或 z，实际为 {axis}")

    value = [0.0, 0.0, 0.0]
    value[axis_to_index[axis]] = math.radians(angle_degrees)
    return tuple(value)



def _prepare_mp4_frame(image: np.ndarray) -> np.ndarray:
    """将帧转换为 RGB，并裁剪为 H.264 可编码的偶数宽高。"""

    rgb = image[..., :3]
    height, width = rgb.shape[:2]
    even_height = height - height % 2
    even_width = width - width % 2
    return np.ascontiguousarray(rgb[:even_height, :even_width])

def _create_video_writers(
    stack: ExitStack,
    out_dir: Path,
    output_format: str,
    fps: int,
):
    """根据命令行参数创建 GIF 和 MP4 写入器。"""

    gif_path = out_dir / "lbs_animation.gif" if output_format in {"gif", "both"} else None
    mp4_path = out_dir / "lbs_animation.mp4" if output_format in {"mp4", "both"} else None

    gif_writer = None
    mp4_writer = None

    if gif_path is not None:
        gif_writer = stack.enter_context(
            imageio.get_writer(
                gif_path,
                mode="I",
                duration=1.0 / fps,
                loop=0,
            )
        )

    if mp4_path is not None:
        mp4_writer = stack.enter_context(
            imageio.get_writer(
                mp4_path,
                format="FFMPEG",
                mode="I",
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=None,
            )
        )

    return gif_writer, mp4_writer, gif_path, mp4_path


def generate_joint_animation(
    model,
    betas: torch.Tensor,
    faces: np.ndarray,
    parents: np.ndarray,
    out_dir: Path,
    *,
    joint_id: int,
    axis: str,
    target_angle_degrees: float,
    forward_frames: int,
    fps: int,
    output_format: str,
    ping_pong: bool,
    keep_frames: bool,
) -> AnimationResult:
    """生成单关节旋转帧，并将同一关节的蒙皮权重作为表面颜色。"""

    if joint_id < 1 or joint_id >= model.lbs_weights.shape[1]:
        raise ValueError(
            f"动画关节编号应位于 [1, {model.lbs_weights.shape[1] - 1}]，"
            f"实际为 {joint_id}"
        )
    if fps <= 0:
        raise ValueError("animation_fps 必须为正整数")
    if output_format not in {"gif", "mp4", "both"}:
        raise ValueError("animation_format 应为 gif、mp4 或 both")

    out_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = out_dir / "animation_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    angle_sequence = build_angle_sequence(
        target_angle_degrees,
        forward_frames,
        ping_pong,
    )
    joint_name = (
        SMPL_JOINT_NAMES[joint_id]
        if joint_id < len(SMPL_JOINT_NAMES)
        else f"joint_{joint_id}"
    )
    weight_scalar = to_numpy(model.lbs_weights[:, joint_id])

    device = betas.device
    dtype = betas.dtype
    global_orient = torch.zeros((1, 3), dtype=dtype, device=device)

    frame_paths: list[Path] = []
    with torch.inference_mode():
        for frame_index, angle_degrees in enumerate(angle_sequence):
            body_pose = torch.zeros((1, 23 * 3), dtype=dtype, device=device)
            set_body_joint_pose(
                body_pose,
                joint_id,
                axis_angle_from_degrees(axis, float(angle_degrees)),
            )
            result = compute_manual_lbs(model, betas, global_orient, body_pose)

            frame_path = frame_dir / f"frame_{frame_index:04d}.png"
            save_animation_frame(
                frame_path,
                to_numpy(result.verts[0]),
                faces,
                to_numpy(result.J_transformed[0]),
                parents,
                title=(
                    f"LBS Animation: {joint_name} "
                    f"({axis}-axis, {angle_degrees:+.1f} deg)"
                ),
                vertex_scalar=weight_scalar,
            )
            frame_paths.append(frame_path)

    with ExitStack() as stack:
        gif_writer, mp4_writer, gif_path, mp4_path = _create_video_writers(
            stack,
            out_dir,
            output_format,
            fps,
        )

        for frame_path in frame_paths:
            image = imageio.imread(frame_path)
            if gif_writer is not None:
                gif_writer.append_data(image)
            if mp4_writer is not None:
                mp4_writer.append_data(_prepare_mp4_frame(image))

    returned_frame_dir: Path | None = frame_dir
    if not keep_frames:
        shutil.rmtree(frame_dir)
        returned_frame_dir = None

    return AnimationResult(
        frame_count=len(frame_paths),
        frame_dir=returned_frame_dir,
        gif_path=gif_path,
        mp4_path=mp4_path,
    )
