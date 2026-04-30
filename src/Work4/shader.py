# src/Work4/shader.py

import taichi as ti

from .config import (
    CAMERA_POS,
    ENABLE_BLINN_PHONG,
    ENABLE_HARD_SHADOW,
    EPS,
    INF,
    LIGHT_COLOR,
    LIGHT_POS,
    T_MIN,
)
from .geometry import intersect_cone, intersect_sphere, vec3_from_tuple


@ti.func
def clamp_color(color: ti.types.vector(3, ti.f32)):
    """
    将 RGB 颜色限制在 [0, 1] 合法范围内，避免过曝发白。
    """
    return ti.Vector(
        [
            ti.min(ti.max(color.x, 0.0), 1.0),
            ti.min(ti.max(color.y, 0.0), 1.0),
            ti.min(ti.max(color.z, 0.0), 1.0),
        ]
    )


@ti.func
def in_shadow(
    hit_pos: ti.types.vector(3, ti.f32),
    normal: ti.types.vector(3, ti.f32),
):
    """
    硬阴影判断。

    从当前交点沿光源方向发射一条 shadow ray。
    如果在到达光源之前再次击中场景中的其他几何体，则当前点处于阴影中。
    """
    shadow = 0

    light_pos = vec3_from_tuple(LIGHT_POS)
    shadow_origin = hit_pos + normal * EPS * 10.0
    light_vec = light_pos - shadow_origin
    light_distance = light_vec.norm()
    shadow_dir = light_vec.normalized()

    sphere_hit, sphere_t, _ = intersect_sphere(shadow_origin, shadow_dir)
    cone_hit, cone_t, _ = intersect_cone(shadow_origin, shadow_dir)

    if sphere_hit == 1 and sphere_t > T_MIN and sphere_t < light_distance:
        shadow = 1

    if cone_hit == 1 and cone_t > T_MIN and cone_t < light_distance:
        shadow = 1

    return shadow


@ti.func
def phong_shader(
    hit_pos: ti.types.vector(3, ti.f32),
    normal: ti.types.vector(3, ti.f32),
    object_color: ti.types.vector(3, ti.f32),
    ka: ti.f32,
    kd: ti.f32,
    ks: ti.f32,
    shininess: ti.f32,
):
    """
    Phong / Blinn-Phong 光照计算。

    标准 Phong：
        I = I_ambient + I_diffuse + I_specular

    Ambient:
        Ka * C_light * C_object

    Diffuse:
        Kd * max(0, N · L) * C_light * C_object

    Specular:
        Ks * max(0, R · V)^n * C_light

    若开启 Blinn-Phong：
        使用半程向量 H 替代反射向量 R，
        Specular = Ks * max(0, N · H)^n * C_light
    """
    light_pos = vec3_from_tuple(LIGHT_POS)
    camera_pos = vec3_from_tuple(CAMERA_POS)
    light_color = vec3_from_tuple(LIGHT_COLOR)

    n = normal.normalized()
    l = (light_pos - hit_pos).normalized()
    v = (camera_pos - hit_pos).normalized()

    ambient = ka * light_color * object_color

    ndotl = ti.max(0.0, n.dot(l))
    diffuse = kd * ndotl * light_color * object_color

    specular_strength = 0.0

    if ndotl > 0.0:
        if ti.static(ENABLE_BLINN_PHONG):
            h = (l + v).normalized()
            specular_strength = ti.pow(ti.max(0.0, n.dot(h)), shininess)
        else:
            r = (2.0 * n.dot(l) * n - l).normalized()
            specular_strength = ti.pow(ti.max(0.0, r.dot(v)), shininess)

    specular = ks * specular_strength * light_color

    color = ambient + diffuse + specular

    if ti.static(ENABLE_HARD_SHADOW):
        if in_shadow(hit_pos, n) == 1:
            color = ambient

    return clamp_color(color)