# src/Work3/geometry.py

from __future__ import annotations

import numpy as np

from .config import NUM_SEGMENTS


def de_casteljau(points: list[np.ndarray], t: float) -> np.ndarray:
    """
    使用 De Casteljau 算法计算 Bézier 曲线在参数 t 处的点。

    参数：
        points: 控制点列表，每个控制点为 np.ndarray([x, y])，坐标范围为 [0, 1]
        t: 曲线参数，范围为 [0, 1]

    返回：
        np.ndarray([x, y])
    """
    if len(points) == 0:
        raise ValueError("de_casteljau() 至少需要 1 个控制点")

    current_points = np.array(points, dtype=np.float32)

    while len(current_points) > 1:
        next_points = []
        for i in range(len(current_points) - 1):
            interpolated = (1.0 - t) * current_points[i] + t * current_points[i + 1]
            next_points.append(interpolated)
        current_points = np.array(next_points, dtype=np.float32)

    return current_points[0]


def generate_bezier_curve_points(
    control_points: list[np.ndarray],
    num_segments: int = NUM_SEGMENTS,
) -> np.ndarray:
    """
    批量生成 Bézier 曲线采样点。

    注意：
        这里在 CPU 端一次性计算全部采样点，
        后续通过 from_numpy() 批量发送到 GPU，避免逐点 CPU-GPU 通信。
    """
    if len(control_points) < 2:
        return np.empty((0, 2), dtype=np.float32)

    curve_points = np.zeros((num_segments + 1, 2), dtype=np.float32)

    for i in range(num_segments + 1):
        t = i / num_segments
        curve_points[i] = de_casteljau(control_points, t)

    return curve_points


def uniform_cubic_bspline_point(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    u: float,
) -> np.ndarray:
    """
    使用均匀三次 B 样条矩阵形式计算局部参数 u 处的点。

    每 4 个相邻控制点构成一段曲线：
        P(u) = 1/6 * [
            (-u^3 + 3u^2 - 3u + 1) P0
            + (3u^3 - 6u^2 + 4) P1
            + (-3u^3 + 3u^2 + 3u + 1) P2
            + u^3 P3
        ]

    参数：
        u: 局部参数，范围为 [0, 1]
    """
    u2 = u * u
    u3 = u2 * u

    b0 = (-u3 + 3.0 * u2 - 3.0 * u + 1.0) / 6.0
    b1 = (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0
    b2 = (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0
    b3 = u3 / 6.0

    return b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3


def generate_uniform_cubic_bspline_points(
    control_points: list[np.ndarray],
    num_segments: int = NUM_SEGMENTS,
) -> np.ndarray:
    """
    生成均匀三次 B 样条曲线采样点。

    说明：
        若控制点数量为 n，则共有 n - 3 段三次 B 样条曲线。
        为了继续复用固定大小的 curve_points_field，
        这里将总采样数量控制为 num_segments + 1，而不是每段都采样 1001 个点。
    """
    if len(control_points) < 4:
        return np.empty((0, 2), dtype=np.float32)

    segment_count = len(control_points) - 3
    curve_points = np.zeros((num_segments + 1, 2), dtype=np.float32)

    for i in range(num_segments + 1):
        global_u = (i / num_segments) * segment_count

        segment_index = int(global_u)
        if segment_index >= segment_count:
            segment_index = segment_count - 1
            local_u = 1.0
        else:
            local_u = global_u - segment_index

        p0 = control_points[segment_index]
        p1 = control_points[segment_index + 1]
        p2 = control_points[segment_index + 2]
        p3 = control_points[segment_index + 3]

        curve_points[i] = uniform_cubic_bspline_point(p0, p1, p2, p3, local_u)

    return curve_points