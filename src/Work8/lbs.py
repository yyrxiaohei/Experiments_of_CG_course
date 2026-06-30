"""手写复现 SMPL 的 Linear Blend Skinning 各阶段。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from smplx.lbs import (
    batch_rigid_transform,
    batch_rodrigues,
    blend_shapes,
    vertices2joints,
)

from .config import DEMO_BETA_VALUES, DEMO_BODY_POSE


@dataclass
class ManualLBSResult:
    """LBS 四阶段及其关键中间量。"""

    # (a) 模板网格
    v_template: torch.Tensor
    J_template: torch.Tensor

    # (b) 形状校正与关节回归
    v_shaped: torch.Tensor
    J: torch.Tensor

    # (c) 姿态校正
    rot_mats: torch.Tensor
    pose_feature: torch.Tensor
    pose_offsets: torch.Tensor
    v_posed: torch.Tensor

    # (d) 刚体层级变换与最终 LBS
    J_transformed: torch.Tensor
    joint_transforms: torch.Tensor
    vertex_transforms: torch.Tensor
    verts: torch.Tensor


@dataclass(frozen=True)
class ErrorMetrics:
    """手写 LBS 与官方前向输出之间的误差。"""

    mean_absolute_error: float
    max_absolute_error: float
    root_mean_square_error: float
    max_vertex_l2_error: float


def build_demo_shape(
    device: torch.device,
    dtype: torch.dtype,
    num_betas: int,
) -> torch.Tensor:
    """构造实验任务 3 使用的非零形状参数。"""

    if num_betas <= 0:
        raise ValueError("num_betas 必须为正整数")

    betas = torch.zeros((1, num_betas), dtype=dtype, device=device)
    for index, value in enumerate(DEMO_BETA_VALUES[:num_betas]):
        betas[0, index] = value
    return betas


def set_body_joint_pose(
    body_pose: torch.Tensor,
    joint_id: int,
    axis_angle: tuple[float, float, float] | torch.Tensor,
) -> None:
    """给 body_pose 中的某个 SMPL 关节写入轴角参数。"""

    if joint_id < 1 or joint_id > 23:
        raise ValueError(f"body_pose 关节编号应位于 [1, 23]，实际为 {joint_id}")

    start = (joint_id - 1) * 3
    value = torch.as_tensor(
        axis_angle,
        dtype=body_pose.dtype,
        device=body_pose.device,
    )
    if value.shape != (3,):
        raise ValueError(f"轴角向量必须包含 3 个分量，实际形状为 {tuple(value.shape)}")
    body_pose[:, start:start + 3] = value


def build_demo_pose(
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造任务 4 和任务 5 使用的非零姿态。"""

    global_orient = torch.zeros((1, 3), dtype=dtype, device=device)
    body_pose = torch.zeros((1, 23 * 3), dtype=dtype, device=device)

    for joint_id, axis_angle in DEMO_BODY_POSE.items():
        set_body_joint_pose(body_pose, joint_id, axis_angle)

    return global_orient, body_pose


def prepare_posedirs(posedirs: torch.Tensor, expected_pose_dim: int) -> torch.Tensor:
    """统一 posedirs 为官方 ``[pose_dim, V * 3]`` 布局。"""

    if posedirs.ndim != 2:
        posedirs = posedirs.reshape(posedirs.shape[0], -1)

    if posedirs.shape[0] == expected_pose_dim:
        return posedirs
    if posedirs.shape[1] == expected_pose_dim:
        return posedirs.transpose(0, 1)

    raise RuntimeError(
        "posedirs 与 pose_feature 维度不匹配："
        f"posedirs.shape={tuple(posedirs.shape)}, "
        f"expected_pose_dim={expected_pose_dim}"
    )


def _expand_template(v_template: torch.Tensor, batch_size: int) -> torch.Tensor:
    """将模板顶点扩展为 ``[B, V, 3]``。"""

    if v_template.ndim == 2:
        return v_template.unsqueeze(0).expand(batch_size, -1, -1)
    if v_template.ndim == 3 and v_template.shape[0] in {1, batch_size}:
        return v_template.expand(batch_size, -1, -1)
    raise RuntimeError(f"无法处理 v_template 形状：{tuple(v_template.shape)}")


def compute_manual_lbs(
    model,
    betas: torch.Tensor,
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
) -> ManualLBSResult:
    """逐步复现官方 ``smplx.lbs.lbs`` 的核心计算过程。"""

    if betas.ndim != 2:
        raise ValueError(f"betas 应为 [B, num_betas]，实际为 {tuple(betas.shape)}")
    if global_orient.shape[-1] != 3:
        raise ValueError("global_orient 的最后一维必须为 3")
    if body_pose.shape[-1] != 23 * 3:
        raise ValueError("SMPL body_pose 的最后一维必须为 69")

    batch_size = max(betas.shape[0], global_orient.shape[0], body_pose.shape[0])
    if any(tensor.shape[0] not in {1, batch_size} for tensor in (betas, global_orient, body_pose)):
        raise ValueError("betas、global_orient 与 body_pose 的批大小不兼容")

    betas = betas.expand(batch_size, -1)
    global_orient = global_orient.expand(batch_size, -1)
    body_pose = body_pose.expand(batch_size, -1)

    device = betas.device
    dtype = betas.dtype
    v_template = _expand_template(model.v_template, batch_size)

    # (b) T_shape = T_bar + B_S(beta)
    shapedirs = model.shapedirs[:, :, :betas.shape[1]]
    shape_offsets = blend_shapes(betas, shapedirs)
    v_shaped = v_template + shape_offsets

    # J(beta) = J_regressor(v_shaped)
    J = vertices2joints(model.J_regressor, v_shaped)

    # (c) T_P(beta, theta) = T_shape + B_P(theta)
    full_pose = torch.cat([global_orient, body_pose], dim=1)
    rot_mats = batch_rodrigues(full_pose.reshape(-1, 3)).reshape(
        batch_size, -1, 3, 3
    )

    identity = torch.eye(3, dtype=dtype, device=device)
    pose_feature = (rot_mats[:, 1:] - identity).reshape(batch_size, -1)
    posedirs = prepare_posedirs(model.posedirs, pose_feature.shape[1])
    pose_offsets = torch.matmul(pose_feature, posedirs).reshape(batch_size, -1, 3)
    v_posed = v_shaped + pose_offsets

    # (d) 运动学链上的全局关节变换。
    J_transformed, joint_transforms = batch_rigid_transform(
        rot_mats,
        J,
        model.parents,
        dtype=dtype,
    )

    num_joints = J.shape[1]
    lbs_weights = model.lbs_weights.unsqueeze(0).expand(batch_size, -1, -1)
    vertex_transforms = torch.matmul(
        lbs_weights,
        joint_transforms.reshape(batch_size, num_joints, 16),
    ).reshape(batch_size, -1, 4, 4)

    homogeneous_one = torch.ones(
        (batch_size, v_posed.shape[1], 1),
        dtype=dtype,
        device=device,
    )
    v_posed_homo = torch.cat([v_posed, homogeneous_one], dim=2)
    v_homo = torch.matmul(vertex_transforms, v_posed_homo.unsqueeze(-1))
    verts = v_homo[:, :, :3, 0]

    J_template = vertices2joints(model.J_regressor, v_template)

    return ManualLBSResult(
        v_template=v_template,
        J_template=J_template,
        v_shaped=v_shaped,
        J=J,
        rot_mats=rot_mats,
        pose_feature=pose_feature,
        pose_offsets=pose_offsets,
        v_posed=v_posed,
        J_transformed=J_transformed,
        joint_transforms=joint_transforms,
        vertex_transforms=vertex_transforms,
        verts=verts,
    )


def compare_with_official_forward(
    model,
    betas: torch.Tensor,
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    manual_verts: torch.Tensor,
) -> ErrorMetrics:
    """使用相同参数调用官方前向，并计算逐顶点误差。"""

    with torch.inference_mode():
        official_output = model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            return_verts=True,
        )

    difference = manual_verts - official_output.vertices
    absolute_difference = difference.abs()
    vertex_l2 = torch.linalg.vector_norm(difference, dim=-1)

    return ErrorMetrics(
        mean_absolute_error=float(absolute_difference.mean().item()),
        max_absolute_error=float(absolute_difference.max().item()),
        root_mean_square_error=float(torch.sqrt(torch.mean(difference.square())).item()),
        max_vertex_l2_error=float(vertex_l2.max().item()),
    )
