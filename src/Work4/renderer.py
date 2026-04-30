# src/Work4/renderer.py

import taichi as ti

from .config import (
    BACKGROUND_COLOR,
    CAMERA_POS,
    CONE_COLOR,
    DEFAULT_KA,
    DEFAULT_KD,
    DEFAULT_KS,
    DEFAULT_SHININESS,
    INF,
    SCREEN_Z,
    SPHERE_COLOR,
    VIEW_HALF_HEIGHT,
    VIEW_HALF_WIDTH,
    WINDOW_HEIGHT,
    WINDOW_RES,
    WINDOW_WIDTH,
)
from .geometry import intersect_cone, intersect_sphere, vec3_from_tuple
from .shader import phong_shader


# 屏幕像素缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=WINDOW_RES)

# Phong 参数 Field，用于和 UI 滑动条实时绑定
ka_field = ti.field(dtype=ti.f32, shape=())
kd_field = ti.field(dtype=ti.f32, shape=())
ks_field = ti.field(dtype=ti.f32, shape=())
shininess_field = ti.field(dtype=ti.f32, shape=())


def init_material_params():
    """
    初始化 Phong 光照参数。
    """
    ka_field[None] = DEFAULT_KA
    kd_field[None] = DEFAULT_KD
    ks_field[None] = DEFAULT_KS
    shininess_field[None] = DEFAULT_SHININESS


def set_material_params(
    ka: float,
    kd: float,
    ks: float,
    shininess: float,
):
    """
    从 Python 端更新 Taichi Field 中的材质参数。
    """
    ka_field[None] = ka
    kd_field[None] = kd
    ks_field[None] = ks
    shininess_field[None] = shininess


@ti.kernel
def render_scene():
    """
    主渲染 Kernel。

    对每一个像素执行：
        1. 由摄像机向成像平面发射一条射线；
        2. 分别计算射线与球体、圆锥的交点；
        3. 执行类似 Z-buffer 的深度竞争，选择最近的正交点；
        4. 对最近交点执行 Phong 着色；
        5. 未命中物体时填充背景色。
    """
    camera_pos = vec3_from_tuple(CAMERA_POS)
    background_color = vec3_from_tuple(BACKGROUND_COLOR)

    sphere_color = vec3_from_tuple(SPHERE_COLOR)
    cone_color = vec3_from_tuple(CONE_COLOR)

    for i, j in pixels:
        # 将像素坐标映射到世界空间中的成像平面。
        u = (ti.cast(i, ti.f32) + 0.5) / ti.cast(WINDOW_WIDTH, ti.f32)
        v = (ti.cast(j, ti.f32) + 0.5) / ti.cast(WINDOW_HEIGHT, ti.f32)

        screen_x = (2.0 * u - 1.0) * VIEW_HALF_WIDTH
        screen_y = (2.0 * v - 1.0) * VIEW_HALF_HEIGHT
        screen_pos = ti.Vector([screen_x, screen_y, SCREEN_Z])

        ray_origin = camera_pos
        ray_dir = (screen_pos - ray_origin).normalized()

        nearest_t = INF
        hit = 0

        hit_pos = ti.Vector([0.0, 0.0, 0.0])
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_color = ti.Vector([0.0, 0.0, 0.0])

        # 1. 球体求交
        sphere_hit, sphere_t, sphere_normal = intersect_sphere(ray_origin, ray_dir)

        if sphere_hit == 1 and sphere_t < nearest_t:
            nearest_t = sphere_t
            hit = 1
            hit_normal = sphere_normal
            hit_color = sphere_color

        # 2. 圆锥求交
        cone_hit, cone_t, cone_normal = intersect_cone(ray_origin, ray_dir)

        if cone_hit == 1 and cone_t < nearest_t:
            nearest_t = cone_t
            hit = 1
            hit_normal = cone_normal
            hit_color = cone_color

        # 3. 最近交点着色
        if hit == 1:
            hit_pos = ray_origin + nearest_t * ray_dir

            pixels[i, j] = phong_shader(
                hit_pos,
                hit_normal,
                hit_color,
                ka_field[None],
                kd_field[None],
                ks_field[None],
                shininess_field[None],
            )
        else:
            pixels[i, j] = background_color