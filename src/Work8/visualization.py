"""SMPL 四阶段静态图与动画帧的 Matplotlib 可视化。"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .config import (
    ANIMATION_DPI,
    CAMERA_AZIMUTH,
    CAMERA_ELEVATION,
    CAMERA_FOCAL_LENGTH,
    DEFAULT_COLORMAP,
    DEFAULT_MESH_RGBA,
    FIGURE_DPI,
    POSE_OFFSET_COLORMAP,
)


def to_numpy(value) -> np.ndarray:
    """安全地把 Torch Tensor 或普通数组转换为 NumPy 数组。"""

    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def smpl_to_plot_coords(points: np.ndarray) -> np.ndarray:
    """将 SMPL 的 ``(x, y, z)`` 转成 Matplotlib 中竖直轴为 z 的坐标。"""

    points = np.asarray(points)
    return points[:, [0, 2, 1]]


def set_axes_equal(ax, vertices: np.ndarray) -> None:
    """设置三轴等比例，避免人体网格显示时被压扁或拉长。"""

    minimums = vertices.min(axis=0)
    maximums = vertices.max(axis=0)
    center = 0.5 * (minimums + maximums)
    radius = 0.5 * max(float(np.max(maximums - minimums)), 1e-6)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def normalize_scalar(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """归一化顶点标量，同时返回原始最小值与最大值。"""

    scalar = np.asarray(values, dtype=np.float64)
    minimum = float(np.min(scalar))
    maximum = float(np.max(scalar))
    normalized = (scalar - minimum) / (maximum - minimum + 1e-12)
    return normalized, minimum, maximum


def get_face_colors_from_vertex_scalar(
    vertex_scalar: np.ndarray,
    faces: np.ndarray,
    cmap_name: str = DEFAULT_COLORMAP,
) -> np.ndarray:
    """把顶点标量平均到面片，并映射为 RGBA 颜色。"""

    normalized, _, _ = normalize_scalar(vertex_scalar)
    face_scalar = normalized[faces].mean(axis=1)
    return plt.get_cmap(cmap_name)(face_scalar)


def get_face_colors_from_joint_weights(
    lbs_weights: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """使用色相表示主导关节，使用明暗表示主导权重强度。"""

    face_weights = lbs_weights[faces].mean(axis=1)
    dominant_joint = np.argmax(face_weights, axis=1)
    dominant_weight = np.max(face_weights, axis=1)

    num_joints = lbs_weights.shape[1]
    palette = plt.get_cmap("hsv")(
        np.linspace(0.0, 1.0, num_joints, endpoint=False)
    )
    face_colors = palette[dominant_joint].copy()

    strength = 0.35 + 0.65 * dominant_weight
    face_colors[:, :3] *= strength[:, None]
    face_colors[:, :3] += (1.0 - strength[:, None]) * 0.88
    face_colors[:, 3] = 1.0
    return face_colors


def shade_face_colors(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
) -> np.ndarray:
    """基于面法向添加简单方向光照，使人体体积更清楚。"""

    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12

    light_direction = np.array([-0.25, -0.55, 0.80], dtype=np.float64)
    light_direction /= np.linalg.norm(light_direction)
    diffuse = np.abs(normals @ light_direction)
    intensity = 0.42 + 0.58 * diffuse

    shaded = np.asarray(face_colors, dtype=np.float64).copy()
    shaded[:, :3] *= intensity[:, None]
    return np.clip(shaded, 0.0, 1.0)


def draw_skeleton(
    ax,
    plot_joints: np.ndarray,
    parents: np.ndarray | None,
) -> None:
    """绘制关节点及其运动学树连线。"""

    ax.scatter(
        plot_joints[:, 0],
        plot_joints[:, 1],
        plot_joints[:, 2],
        c="white",
        s=14,
        depthshade=False,
        edgecolors="black",
        linewidths=0.35,
    )

    if parents is None:
        return

    parents = np.asarray(parents, dtype=np.int64).reshape(-1)
    count = min(len(parents), len(plot_joints))
    for joint_id in range(1, count):
        parent_id = int(parents[joint_id])
        if parent_id < 0 or parent_id >= count:
            continue
        segment = plot_joints[[parent_id, joint_id]]
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color="black",
            linewidth=0.8,
            alpha=0.85,
        )


def draw_mesh(
    ax,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    joints: np.ndarray | None = None,
    parents: np.ndarray | None = None,
    vertex_scalar: np.ndarray | None = None,
    face_colors: np.ndarray | None = None,
    cmap_name: str = DEFAULT_COLORMAP,
    title: str = "",
    elev: float = CAMERA_ELEVATION,
    azim: float = CAMERA_AZIMUTH,
) -> None:
    """在指定三维坐标轴中绘制网格、关节和骨架。"""

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    plot_vertices = smpl_to_plot_coords(vertices)

    if face_colors is not None:
        colors = np.asarray(face_colors, dtype=np.float64).copy()
    elif vertex_scalar is None:
        colors = np.tile(DEFAULT_MESH_RGBA[None, :], (faces.shape[0], 1))
    else:
        colors = get_face_colors_from_vertex_scalar(
            vertex_scalar,
            faces,
            cmap_name=cmap_name,
        )

    colors = shade_face_colors(plot_vertices, faces, colors)
    mesh = Poly3DCollection(
        plot_vertices[faces],
        facecolors=colors,
        linewidths=0.03,
        edgecolors=(0.0, 0.0, 0.0, 0.05),
    )
    ax.add_collection3d(mesh)

    if joints is not None:
        draw_skeleton(
            ax,
            smpl_to_plot_coords(np.asarray(joints, dtype=np.float64)),
            parents,
        )

    set_axes_equal(ax, plot_vertices)
    try:
        ax.set_proj_type("persp", focal_length=CAMERA_FOCAL_LENGTH)
    except TypeError:
        ax.set_proj_type("persp")
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)


def add_scalar_colorbar(
    fig,
    ax,
    scalar: np.ndarray,
    cmap_name: str,
    label: str,
) -> None:
    """给权重或位移标量图添加颜色条。"""

    scalar = np.asarray(scalar, dtype=np.float64)
    minimum = float(np.min(scalar))
    maximum = float(np.max(scalar))
    if abs(maximum - minimum) < 1e-12:
        maximum = minimum + 1e-12

    mappable = ScalarMappable(
        norm=Normalize(vmin=minimum, vmax=maximum),
        cmap=plt.get_cmap(cmap_name),
    )
    mappable.set_array([])
    colorbar = fig.colorbar(mappable, ax=ax, fraction=0.035, pad=0.01)
    colorbar.set_label(label, fontsize=8)
    colorbar.ax.tick_params(labelsize=7)


def save_single_figure(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    joints: np.ndarray | None = None,
    parents: np.ndarray | None = None,
    vertex_scalar: np.ndarray | None = None,
    cmap_name: str = DEFAULT_COLORMAP,
    scalar_label: str | None = None,
    title: str = "",
    dpi: int = FIGURE_DPI,
) -> None:
    """保存一张独立阶段图或动画帧。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(5.2, 6.2))
    axis = figure.add_subplot(111, projection="3d")
    draw_mesh(
        axis,
        vertices,
        faces,
        joints=joints,
        parents=parents,
        vertex_scalar=vertex_scalar,
        cmap_name=cmap_name,
        title=title,
    )

    if vertex_scalar is not None and scalar_label:
        add_scalar_colorbar(
            figure,
            axis,
            vertex_scalar,
            cmap_name,
            scalar_label,
        )

    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_comparison_grid(
    path: Path,
    data: Mapping[str, np.ndarray],
    faces: np.ndarray,
    parents: np.ndarray,
) -> None:
    """保存实验要求的 2×2 四阶段总对比图。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(14, 10))

    axis_a = figure.add_subplot(221, projection="3d")
    draw_mesh(
        axis_a,
        data["v_template"],
        faces,
        joints=data["J_template"],
        parents=parents,
        vertex_scalar=data["weight_scalar"],
        title="(a) Template + LBS Weights",
    )

    axis_b = figure.add_subplot(222, projection="3d")
    draw_mesh(
        axis_b,
        data["v_shaped"],
        faces,
        joints=data["J"],
        parents=parents,
        title="(b) Shape Blend + Joint Regression",
    )

    axis_c = figure.add_subplot(223, projection="3d")
    draw_mesh(
        axis_c,
        data["v_posed"],
        faces,
        joints=data["J"],
        parents=parents,
        vertex_scalar=data["pose_offset_norm"],
        cmap_name=POSE_OFFSET_COLORMAP,
        title="(c) Pose Blend Shapes",
    )

    axis_d = figure.add_subplot(224, projection="3d")
    draw_mesh(
        axis_d,
        data["verts"],
        faces,
        joints=data["J_transformed"],
        parents=parents,
        title="(d) Final Skinned Mesh",
    )

    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def save_all_joint_weights_figure(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    parents: np.ndarray,
    lbs_weights: np.ndarray,
) -> None:
    """保存全关节主导权重分布图。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(7, 8))
    axis = figure.add_subplot(111, projection="3d")
    draw_mesh(
        axis,
        vertices,
        faces,
        joints=joints,
        parents=parents,
        face_colors=get_face_colors_from_joint_weights(lbs_weights, faces),
        title="All Joint Dominant LBS Weights",
    )

    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def save_animation_frame(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    parents: np.ndarray,
    title: str,
    vertex_scalar: np.ndarray | None = None,
) -> None:
    """以较低 DPI 保存选做动画的一帧。"""

    save_single_figure(
        path,
        vertices,
        faces,
        joints=joints,
        parents=parents,
        vertex_scalar=vertex_scalar,
        title=title,
        dpi=ANIMATION_DPI,
    )
