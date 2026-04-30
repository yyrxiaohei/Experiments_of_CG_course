# src/Work3/main.py

from __future__ import annotations

import numpy as np
import taichi as ti

# 注意：Taichi 初始化必须尽量靠前执行
ti.init(arch=ti.gpu)

from .config import (  # noqa: E402
    CONTROL_POINT_COLOR,
    CONTROL_POINT_RADIUS,
    CONTROL_POLYGON_COLOR,
    CONTROL_POLYGON_WIDTH,
    ENABLE_ANTI_ALIASING,
    HIDDEN_POS,
    MAX_CONTROL_LINE_VERTICES,
    MAX_CONTROL_POINTS,
    MAX_CURVE_POINTS,
    MIN_BEZIER_CONTROL_POINTS,
    MIN_BSPLINE_CONTROL_POINTS,
    MODE_BEZIER,
    MODE_BSPLINE,
    NUM_SEGMENTS,
    WINDOW_RES,
    WINDOW_TITLE,
)
from .geometry import (  # noqa: E402
    generate_bezier_curve_points,
    generate_uniform_cubic_bspline_points,
)
from .rasterizer import (  # noqa: E402
    clear_pixels,
    control_line_vertices,
    curve_points_field,
    draw_curve_kernel,
    gui_points,
    pixels,
    reset_gui_object_pools,
)


def build_gui_points_pool(control_points: list[np.ndarray]) -> np.ndarray:
    """
    构造控制点对象池。

    canvas.circles() 接收固定长度 Field。
    因此这里创建固定长度 NumPy 数组，未使用位置填入屏幕外坐标。
    """
    points_pool = np.full(
        (MAX_CONTROL_POINTS, 2),
        HIDDEN_POS,
        dtype=np.float32,
    )

    for i, point in enumerate(control_points[:MAX_CONTROL_POINTS]):
        points_pool[i] = point

    return points_pool


def build_control_polygon_pool(control_points: list[np.ndarray]) -> np.ndarray:
    """
    构造控制多边形线段对象池。

    canvas.lines() 会按 0-1、2-3、4-5 的方式绘制线段。
    因此 n 个控制点需要 n - 1 条线段，即 2 * (n - 1) 个线段端点。
    """
    line_pool = np.full(
        (MAX_CONTROL_LINE_VERTICES, 2),
        HIDDEN_POS,
        dtype=np.float32,
    )

    if len(control_points) < 2:
        return line_pool

    line_index = 0
    for i in range(len(control_points) - 1):
        if line_index + 1 >= MAX_CONTROL_LINE_VERTICES:
            break

        line_pool[line_index] = control_points[i]
        line_pool[line_index + 1] = control_points[i + 1]
        line_index += 2

    return line_pool


def upload_gui_object_pools(control_points: list[np.ndarray]):
    """
    将控制点与控制多边形对象池批量上传到 GPU。
    """
    points_pool = build_gui_points_pool(control_points)
    lines_pool = build_control_polygon_pool(control_points)

    gui_points.from_numpy(points_pool)
    control_line_vertices.from_numpy(lines_pool)


def generate_curve_points(
    control_points: list[np.ndarray],
    curve_mode: int,
) -> np.ndarray:
    """
    根据当前曲线模式生成曲线采样点。
    """
    if curve_mode == MODE_BEZIER:
        if len(control_points) < MIN_BEZIER_CONTROL_POINTS:
            return np.empty((0, 2), dtype=np.float32)

        return generate_bezier_curve_points(
            control_points=control_points,
            num_segments=NUM_SEGMENTS,
        )

    if curve_mode == MODE_BSPLINE:
        if len(control_points) < MIN_BSPLINE_CONTROL_POINTS:
            return np.empty((0, 2), dtype=np.float32)

        return generate_uniform_cubic_bspline_points(
            control_points=control_points,
            num_segments=NUM_SEGMENTS,
        )

    return np.empty((0, 2), dtype=np.float32)


def upload_curve_points(curve_points: np.ndarray) -> int:
    """
    将曲线采样点批量上传到 GPU。

    返回：
        实际上传并需要绘制的点数量。
    """
    if curve_points.size == 0:
        return 0

    point_count = min(len(curve_points), MAX_CURVE_POINTS)

    curve_pool = np.full(
        (MAX_CURVE_POINTS, 2),
        HIDDEN_POS,
        dtype=np.float32,
    )
    curve_pool[:point_count] = curve_points[:point_count]

    curve_points_field.from_numpy(curve_pool)

    return point_count


def get_mode_name(curve_mode: int) -> str:
    """
    获取当前曲线模式名称，用于窗口标题和控制台提示。
    """
    if curve_mode == MODE_BEZIER:
        return "Bezier"

    if curve_mode == MODE_BSPLINE:
        return "Uniform Cubic B-Spline"

    return "Unknown"


def print_help_message():
    """
    打印交互说明。
    """
    print("============================================================")
    print("Experiment 3: Bezier Curve Rasterization")
    print("------------------------------------------------------------")
    print("鼠标左键：添加控制点")
    print("C 键：清空全部控制点")
    print("B 键：切换 Bézier / 均匀三次 B 样条模式")
    print("ESC：退出程序")
    print("------------------------------------------------------------")
    print("说明：")
    print("1. Bézier 模式下，控制点数量 >= 2 时绘制曲线。")
    print("2. B 样条模式下，控制点数量 >= 4 时绘制曲线。")
    print("3. 曲线点在 CPU 批量计算，再一次性上传 GPU 光栅化。")
    print("============================================================")


def main():
    print("正在初始化 GPU Field 与 GGUI 窗口，请稍候...")

    control_points: list[np.ndarray] = []
    curve_mode = MODE_BEZIER

    reset_gui_object_pools()

    window = ti.ui.Window(
        name=WINDOW_TITLE,
        res=WINDOW_RES,
        vsync=True,
    )
    canvas = window.get_canvas()

    print_help_message()
    print("初始化完成！请在窗口中点击鼠标左键添加控制点。")

    while window.running:
        # =========================
        # 事件处理
        # =========================
        while window.get_event(ti.ui.PRESS):
            event_key = window.event.key

            if event_key == ti.ui.ESCAPE:
                window.running = False

            elif event_key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    mouse_x, mouse_y = window.get_cursor_pos()
                    control_points.append(
                        np.array([mouse_x, mouse_y], dtype=np.float32)
                    )

            elif event_key == "c" or event_key == "C":
                control_points.clear()
                reset_gui_object_pools()
                print("已清空控制点。")

            elif event_key == "b" or event_key == "B":
                if curve_mode == MODE_BEZIER:
                    curve_mode = MODE_BSPLINE
                else:
                    curve_mode = MODE_BEZIER

                print(f"当前曲线模式：{get_mode_name(curve_mode)}")

        # =========================
        # 清空像素缓冲区
        # =========================
        clear_pixels()

        # =========================
        # CPU 批量计算曲线采样点
        # =========================
        curve_points = generate_curve_points(
            control_points=control_points,
            curve_mode=curve_mode,
        )

        # =========================
        # 批量上传曲线点，并由 GPU 光栅化绘制
        # =========================
        point_count = upload_curve_points(curve_points)

        if point_count > 0:
            draw_curve_kernel(
                point_count,
                1 if ENABLE_ANTI_ALIASING else 0,
            )

        # =========================
        # 批量上传控制点与控制多边形对象池
        # =========================
        upload_gui_object_pools(control_points)

        # =========================
        # GGUI 显示
        # =========================
        canvas.set_image(pixels)

        canvas.lines(
            control_line_vertices,
            width=CONTROL_POLYGON_WIDTH,
            color=CONTROL_POLYGON_COLOR,
        )

        canvas.circles(
            gui_points,
            radius=CONTROL_POINT_RADIUS,
            color=CONTROL_POINT_COLOR,
        )

        window.show()


if __name__ == "__main__":
    main()