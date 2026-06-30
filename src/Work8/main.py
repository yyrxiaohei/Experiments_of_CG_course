"""实验八入口：生成 LBS 四阶段结果、误差摘要和单关节动画。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .animation import AnimationResult, generate_joint_animation
from .config import (
    DEFAULT_ANIMATION_ANGLE_DEGREES,
    DEFAULT_ANIMATION_AXIS,
    DEFAULT_ANIMATION_FORMAT,
    DEFAULT_ANIMATION_FORWARD_FRAMES,
    DEFAULT_ANIMATION_FPS,
    DEFAULT_ANIMATION_JOINT_ID,
    DEFAULT_ANIMATION_PING_PONG,
    DEFAULT_KEEP_ANIMATION_FRAMES,
    DEFAULT_MODEL_DIR,
    DEFAULT_NUM_BETAS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WEIGHT_JOINT_ID,
    POSE_OFFSET_COLORMAP,
    SMPL_JOINT_NAMES,
    WORK8_DIR,
)
from .lbs import (
    ErrorMetrics,
    ManualLBSResult,
    build_demo_pose,
    build_demo_shape,
    compare_with_official_forward,
    compute_manual_lbs,
)
from .model_utils import (
    ModelInfo,
    choose_device,
    collect_model_info,
    load_smpl_model,
    resolve_path,
)
from .visualization import (
    save_all_joint_weights_figure,
    save_comparison_grid,
    save_single_figure,
    to_numpy,
)


def validate_joint_id(joint_id: int, num_joints: int, argument_name: str) -> None:
    """检查权重可视化或动画使用的关节编号。"""

    if joint_id < 0 or joint_id >= num_joints:
        raise ValueError(
            f"{argument_name} 越界：{joint_id}，可选范围为 [0, {num_joints - 1}]"
        )


def joint_display_name(joint_id: int) -> str:
    """返回适合写入标题和摘要的关节名称。"""

    if 0 <= joint_id < len(SMPL_JOINT_NAMES):
        return SMPL_JOINT_NAMES[joint_id]
    return f"joint_{joint_id}"


def save_required_figures(
    out_dir: Path,
    model,
    faces: np.ndarray,
    parents: np.ndarray,
    result: ManualLBSResult,
    joint_id: int,
) -> None:
    """保存任务 2 至任务 6 要求的全部图片。"""

    weight_scalar = to_numpy(model.lbs_weights[:, joint_id])
    pose_offset_norm = np.linalg.norm(to_numpy(result.pose_offsets[0]), axis=1)

    v_template = to_numpy(result.v_template[0])
    J_template = to_numpy(result.J_template[0])
    v_shaped = to_numpy(result.v_shaped[0])
    J = to_numpy(result.J[0])
    v_posed = to_numpy(result.v_posed[0])
    verts = to_numpy(result.verts[0])
    J_transformed = to_numpy(result.J_transformed[0])

    save_single_figure(
        out_dir / "stage_a_template_weights.png",
        v_template,
        faces,
        joints=J_template,
        parents=parents,
        vertex_scalar=weight_scalar,
        scalar_label="LBS weight",
        title=(
            f"(a) Template Mesh + Weight of Joint {joint_id} "
            f"({joint_display_name(joint_id)})"
        ),
    )

    save_all_joint_weights_figure(
        out_dir / "all_joint_weights.png",
        v_template,
        faces,
        J_template,
        parents,
        to_numpy(model.lbs_weights),
    )

    save_single_figure(
        out_dir / "stage_b_shaped_joints.png",
        v_shaped,
        faces,
        joints=J,
        parents=parents,
        title="(b) Shape Blend + Joint Regression",
    )

    save_single_figure(
        out_dir / "stage_c_pose_offsets.png",
        v_posed,
        faces,
        joints=J,
        parents=parents,
        vertex_scalar=pose_offset_norm,
        cmap_name=POSE_OFFSET_COLORMAP,
        scalar_label="|pose_offsets|",
        title="(c) Pose Blend Shapes",
    )

    save_single_figure(
        out_dir / "stage_d_lbs_result.png",
        verts,
        faces,
        joints=J_transformed,
        parents=parents,
        title="(d) Final LBS Result",
    )

    save_comparison_grid(
        out_dir / "comparison_grid.png",
        {
            "v_template": v_template,
            "J_template": J_template,
            "v_shaped": v_shaped,
            "J": J,
            "v_posed": v_posed,
            "verts": verts,
            "J_transformed": J_transformed,
            "weight_scalar": weight_scalar,
            "pose_offset_norm": pose_offset_norm,
        },
        faces,
        parents,
    )


def write_summary(
    path: Path,
    model_info: ModelInfo,
    joint_id: int,
    betas: torch.Tensor,
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    metrics: ErrorMetrics,
    device: torch.device,
    animation_result: AnimationResult | None,
    args: argparse.Namespace,
) -> None:
    """保存任务 1、任务 7 和选做部分的运行摘要。"""

    nonzero_pose = torch.nonzero(body_pose[0].reshape(23, 3).abs().sum(dim=1) > 0)
    nonzero_joint_ids = [int(index.item()) + 1 for index in nonzero_pose]

    lines = [
        "===== SMPL LBS Lab Summary =====",
        f"device: {device}",
        f"num_vertices: {model_info.num_vertices}",
        f"num_faces: {model_info.num_faces}",
        f"num_joints(from lbs_weights): {model_info.num_joints}",
        f"num_betas: {model_info.num_betas}",
        f"betas: {to_numpy(betas[0]).tolist()}",
        f"global_orient: {to_numpy(global_orient[0]).tolist()}",
        f"nonzero_body_pose_joint_ids: {nonzero_joint_ids}",
        (
            f"visualized_joint: {joint_id} "
            f"({joint_display_name(joint_id)})"
        ),
        "",
        "===== Manual LBS vs Official Forward =====",
        f"mean_absolute_error: {metrics.mean_absolute_error:.12e}",
        f"max_absolute_error: {metrics.max_absolute_error:.12e}",
        f"root_mean_square_error: {metrics.root_mean_square_error:.12e}",
        f"max_vertex_l2_error: {metrics.max_vertex_l2_error:.12e}",
        "",
        "===== Required Outputs =====",
        "stage_a_template_weights.png",
        "all_joint_weights.png",
        "stage_b_shaped_joints.png",
        "stage_c_pose_offsets.png",
        "stage_d_lbs_result.png",
        "comparison_grid.png",
        "summary.txt",
    ]

    if animation_result is not None:
        lines.extend(
            [
                "",
                "===== Optional Joint Animation =====",
                (
                    f"animation_joint: {args.animation_joint_id} "
                    f"({joint_display_name(args.animation_joint_id)})"
                ),
                f"animation_axis: {args.animation_axis}",
                f"target_angle_degrees: {args.animation_angle}",
                f"forward_frames: {args.animation_frames}",
                f"rendered_frame_count: {animation_result.frame_count}",
                f"fps: {args.animation_fps}",
                f"ping_pong: {not args.no_animation_ping_pong}",
                f"frame_dir: {animation_result.frame_dir}",
                f"gif_path: {animation_result.gif_path}",
                f"mp4_path: {animation_result.mp4_path}",
            ]
        )
    else:
        lines.extend(["", "===== Optional Joint Animation =====", "skipped: true"])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    """执行完整实验流程。"""

    device = choose_device(args.device)
    dtype = torch.float32
    model_dir = resolve_path(args.model_dir, WORK8_DIR)
    out_dir = resolve_path(args.out_dir, WORK8_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_smpl_model(
        model_dir,
        num_betas=args.num_betas,
        device=device,
        dtype=dtype,
    )
    faces = np.asarray(model.faces, dtype=np.int32)
    parents = to_numpy(model.parents).astype(np.int64)
    model_info = collect_model_info(model, faces, args.num_betas)

    validate_joint_id(args.joint_id, model_info.num_joints, "joint_id")
    validate_joint_id(
        args.animation_joint_id,
        model_info.num_joints,
        "animation_joint_id",
    )
    if args.animation_joint_id == 0 and not args.skip_animation:
        raise ValueError("选做动画应驱动 body_pose 关节，animation_joint_id 不能为根关节 0")

    betas = build_demo_shape(device, dtype, args.num_betas)
    global_orient, body_pose = build_demo_pose(device, dtype)

    with torch.inference_mode():
        result = compute_manual_lbs(model, betas, global_orient, body_pose)
        metrics = compare_with_official_forward(
            model,
            betas,
            global_orient,
            body_pose,
            result.verts,
        )

    save_required_figures(
        out_dir,
        model,
        faces,
        parents,
        result,
        args.joint_id,
    )

    animation_result = None
    if not args.skip_animation:
        animation_result = generate_joint_animation(
            model,
            betas,
            faces,
            parents,
            out_dir,
            joint_id=args.animation_joint_id,
            axis=args.animation_axis,
            target_angle_degrees=args.animation_angle,
            forward_frames=args.animation_frames,
            fps=args.animation_fps,
            output_format=args.animation_format,
            ping_pong=not args.no_animation_ping_pong,
            keep_frames=not args.no_save_animation_frames,
        )

    write_summary(
        out_dir / "summary.txt",
        model_info,
        args.joint_id,
        betas,
        global_orient,
        body_pose,
        metrics,
        device,
        animation_result,
        args,
    )

    print("运行完成。")
    print(f"顶点数: {model_info.num_vertices}")
    print(f"面片数: {model_info.num_faces}")
    print(f"关节数: {model_info.num_joints}")
    print(f"betas 维度: {model_info.num_betas}")
    print(f"手写 LBS 平均绝对误差: {metrics.mean_absolute_error:.12e}")
    print(f"手写 LBS 最大绝对误差: {metrics.max_absolute_error:.12e}")
    if animation_result is not None:
        print(f"动画帧数: {animation_result.frame_count}")
        if animation_result.gif_path is not None:
            print(f"GIF: {animation_result.gif_path}")
        if animation_result.mp4_path is not None:
            print(f"MP4: {animation_result.mp4_path}")
    print(f"全部结果已保存到: {out_dir}")


def build_argument_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(
        description="计算机图形学实验八：SMPL Linear Blend Skinning",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="模型目录，默认包含 models/smpl/SMPL_NEUTRAL.pkl",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="图片、动画和 summary.txt 的输出目录",
    )
    parser.add_argument(
        "--joint-id",
        type=int,
        default=DEFAULT_WEIGHT_JOINT_ID,
        help="阶段 (a) 单关节权重热力图使用的关节编号",
    )
    parser.add_argument(
        "--num-betas",
        type=int,
        default=DEFAULT_NUM_BETAS,
        help="使用的 SMPL shape 参数数量",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="Torch 运行设备",
    )

    parser.add_argument(
        "--skip-animation",
        action="store_true",
        help="仅执行必做部分，不生成选做动画",
    )
    parser.add_argument(
        "--animation-joint-id",
        type=int,
        default=DEFAULT_ANIMATION_JOINT_ID,
        help="选做动画中被驱动的 SMPL body_pose 关节编号",
    )
    parser.add_argument(
        "--animation-axis",
        choices=("x", "y", "z"),
        default=DEFAULT_ANIMATION_AXIS,
        help="动画关节的轴角旋转轴",
    )
    parser.add_argument(
        "--animation-angle",
        type=float,
        default=DEFAULT_ANIMATION_ANGLE_DEGREES,
        help="动画目标旋转角度，单位为度",
    )
    parser.add_argument(
        "--animation-frames",
        type=int,
        default=DEFAULT_ANIMATION_FORWARD_FRAMES,
        help="从 0 旋转到目标角度所用的帧数",
    )
    parser.add_argument(
        "--animation-fps",
        type=int,
        default=DEFAULT_ANIMATION_FPS,
        help="GIF/MP4 帧率",
    )
    parser.add_argument(
        "--animation-format",
        choices=("gif", "mp4", "both"),
        default=DEFAULT_ANIMATION_FORMAT,
        help="动画导出格式",
    )
    parser.add_argument(
        "--no-animation-ping-pong",
        action="store_true",
        default=not DEFAULT_ANIMATION_PING_PONG,
        help="只播放 0 到目标角度，不生成反向回程帧",
    )
    parser.add_argument(
        "--no-save-animation-frames",
        action="store_true",
        default=not DEFAULT_KEEP_ANIMATION_FRAMES,
        help="动画合成后删除 animation_frames 目录",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    run(parser.parse_args())


if __name__ == "__main__":
    main()
