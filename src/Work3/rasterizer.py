# src/Work3/rasterizer.py

import taichi as ti

from .config import (
    ANTI_ALIASING_RADIUS,
    BACKGROUND_COLOR,
    CONTROL_POLYGON_COLOR,
    CONTROL_POINT_COLOR,
    CURVE_COLOR,
    HIDDEN_POS,
    MAX_CONTROL_LINE_VERTICES,
    MAX_CONTROL_POINTS,
    MAX_CURVE_POINTS,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)

# =========================
# GPU Field 定义
# =========================

# 模拟屏幕显存：每个像素存储 RGB 三通道颜色
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WINDOW_WIDTH, WINDOW_HEIGHT))

# CPU 批量算好的曲线点，一次性传入该 Field，再交给 GPU 光栅化
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CURVE_POINTS)

# 控制点对象池：固定容量，未使用位置放到屏幕外
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)

# 控制多边形线段对象池：固定容量，未使用位置放到屏幕外
control_line_vertices = ti.Vector.field(
    2,
    dtype=ti.f32,
    shape=MAX_CONTROL_LINE_VERTICES,
)


@ti.kernel
def clear_pixels():
    """
    清空像素缓冲区。
    每一帧先把整张画布恢复为背景色。
    """
    for i, j in pixels:
        pixels[i, j] = ti.Vector(
            [
                BACKGROUND_COLOR[0],
                BACKGROUND_COLOR[1],
                BACKGROUND_COLOR[2],
            ]
        )


@ti.func
def blend_pixel(x: ti.i32, y: ti.i32, color: ti.template(), alpha: ti.f32):
    """
    对指定像素进行 alpha 混合。
    """
    if 0 <= x < WINDOW_WIDTH and 0 <= y < WINDOW_HEIGHT:
        old_color = pixels[x, y]
        pixels[x, y] = old_color * (1.0 - alpha) + color * alpha


@ti.kernel
def draw_curve_kernel(n: ti.i32, enable_anti_aliasing: ti.i32):
    """
    GPU 光栅化曲线采样点。

    参数：
        n: 实际需要绘制的曲线点数量；
        enable_anti_aliasing:
            0 表示基础光栅化；
            1 表示开启 3x3 邻域距离衰减反走样。
    """
    for i in range(n):
        p = curve_points_field[i]

        # 归一化坐标 [0, 1] -> 物理像素坐标 [0, width / height]
        fx = p.x * ti.cast(WINDOW_WIDTH - 1, ti.f32)
        fy = p.y * ti.cast(WINDOW_HEIGHT - 1, ti.f32)

        curve_color = ti.Vector([CURVE_COLOR[0], CURVE_COLOR[1], CURVE_COLOR[2]])

        if enable_anti_aliasing == 0:
            x = ti.cast(fx, ti.i32)
            y = ti.cast(fy, ti.i32)

            if 0 <= x < WINDOW_WIDTH and 0 <= y < WINDOW_HEIGHT:
                pixels[x, y] = curve_color

        else:
            center_x = ti.cast(fx, ti.i32)
            center_y = ti.cast(fy, ti.i32)

            for dx, dy in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                x = center_x + dx
                y = center_y + dy

                pixel_center = ti.Vector(
                    [
                        ti.cast(x, ti.f32) + 0.5,
                        ti.cast(y, ti.f32) + 0.5,
                    ]
                )
                exact_pos = ti.Vector([fx, fy])

                dist = (pixel_center - exact_pos).norm()
                alpha = ti.max(0.0, 1.0 - dist / ANTI_ALIASING_RADIUS)

                blend_pixel(x, y, curve_color, alpha)


@ti.kernel
def reset_gui_object_pools():
    """
    将控制点与控制线对象池全部移动到屏幕外。
    """
    for i in gui_points:
        gui_points[i] = ti.Vector([HIDDEN_POS, HIDDEN_POS])

    for i in control_line_vertices:
        control_line_vertices[i] = ti.Vector([HIDDEN_POS, HIDDEN_POS])