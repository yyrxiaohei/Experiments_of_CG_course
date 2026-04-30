# src/Work4/geometry.py

import taichi as ti

from .config import (
    CONE_APEX,
    CONE_BASE_RADIUS,
    CONE_BASE_Y,
    EPS,
    INF,
    SPHERE_CENTER,
    SPHERE_RADIUS,
    T_MIN,
)


@ti.func
def vec3_from_tuple(data: ti.template()) -> ti.types.vector(3, ti.f32):
    """
    将 Python tuple 常量转换为 Taichi 三维向量。
    """
    return ti.Vector([data[0], data[1], data[2]])


@ti.func
def intersect_sphere(
    ray_origin: ti.types.vector(3, ti.f32),
    ray_dir: ti.types.vector(3, ti.f32),
):
    """
    计算射线与球体的最近正交点。

    球面隐式方程：
        |P - C|^2 = r^2

    射线方程：
        P(t) = O + tD

    返回：
        hit    : 是否击中
        t      : 最近交点距离
        normal : 交点处单位法向量
    """
    hit = 0
    nearest_t = INF
    normal = ti.Vector([0.0, 0.0, 0.0])

    center = vec3_from_tuple(SPHERE_CENTER)
    radius = SPHERE_RADIUS

    oc = ray_origin - center

    a = ray_dir.dot(ray_dir)
    b = 2.0 * oc.dot(ray_dir)
    c = oc.dot(oc) - radius * radius

    discriminant = b * b - 4.0 * a * c

    if discriminant >= 0.0:
        sqrt_discriminant = ti.sqrt(discriminant)

        t1 = (-b - sqrt_discriminant) / (2.0 * a)
        t2 = (-b + sqrt_discriminant) / (2.0 * a)

        if t1 > T_MIN:
            nearest_t = t1
            hit = 1
        elif t2 > T_MIN:
            nearest_t = t2
            hit = 1

        if hit == 1:
            hit_pos = ray_origin + nearest_t * ray_dir
            normal = (hit_pos - center).normalized()

    return hit, nearest_t, normal


@ti.func
def intersect_cone_side(
    ray_origin: ti.types.vector(3, ti.f32),
    ray_dir: ti.types.vector(3, ti.f32),
):
    """
    计算射线与有限圆锥侧面的最近正交点。

    圆锥参数：
        顶点 A = CONE_APEX
        底面高度 y = CONE_BASE_Y
        底面半径 r = CONE_BASE_RADIUS
        轴线方向沿 y 轴向下

    令 q = P - A，则圆锥侧面隐式方程为：
        q.x^2 + q.z^2 - k^2 * q.y^2 = 0

    其中：
        k = r / h
        h = A.y - base_y

    有限高度约束：
        base_y <= P.y <= apex_y
    """
    hit = 0
    nearest_t = INF
    normal = ti.Vector([0.0, 0.0, 0.0])

    apex = vec3_from_tuple(CONE_APEX)
    height = CONE_APEX[1] - CONE_BASE_Y
    k = CONE_BASE_RADIUS / height
    k2 = k * k

    oc = ray_origin - apex

    a = ray_dir.x * ray_dir.x + ray_dir.z * ray_dir.z - k2 * ray_dir.y * ray_dir.y
    b = 2.0 * (oc.x * ray_dir.x + oc.z * ray_dir.z - k2 * oc.y * ray_dir.y)
    c = oc.x * oc.x + oc.z * oc.z - k2 * oc.y * oc.y

    discriminant = b * b - 4.0 * a * c

    if ti.abs(a) > EPS and discriminant >= 0.0:
        sqrt_discriminant = ti.sqrt(discriminant)

        t1 = (-b - sqrt_discriminant) / (2.0 * a)
        t2 = (-b + sqrt_discriminant) / (2.0 * a)

        if t1 > T_MIN:
            p1 = ray_origin + t1 * ray_dir
            if p1.y <= CONE_APEX[1] and p1.y >= CONE_BASE_Y:
                nearest_t = t1
                hit = 1

        if t2 > T_MIN and t2 < nearest_t:
            p2 = ray_origin + t2 * ray_dir
            if p2.y <= CONE_APEX[1] and p2.y >= CONE_BASE_Y:
                nearest_t = t2
                hit = 1

        if hit == 1:
            hit_pos = ray_origin + nearest_t * ray_dir
            q = hit_pos - apex

            # 圆锥隐式函数 F = x^2 + z^2 - k^2 y^2
            # 法向量为梯度 grad(F) = (2x, -2k^2y, 2z)
            normal = ti.Vector([q.x, -k2 * q.y, q.z]).normalized()

    return hit, nearest_t, normal


@ti.func
def intersect_cone_base(
    ray_origin: ti.types.vector(3, ti.f32),
    ray_dir: ti.types.vector(3, ti.f32),
):
    """
    计算射线与圆锥底面圆盘的交点。

    底面圆盘位于：
        y = CONE_BASE_Y

    半径：
        CONE_BASE_RADIUS
    """
    hit = 0
    nearest_t = INF
    normal = ti.Vector([0.0, 0.0, 0.0])

    apex = vec3_from_tuple(CONE_APEX)
    base_center = ti.Vector([apex.x, CONE_BASE_Y, apex.z])

    if ti.abs(ray_dir.y) > EPS:
        t = (CONE_BASE_Y - ray_origin.y) / ray_dir.y

        if t > T_MIN:
            hit_pos = ray_origin + t * ray_dir
            offset = hit_pos - base_center
            dist2 = offset.x * offset.x + offset.z * offset.z

            if dist2 <= CONE_BASE_RADIUS * CONE_BASE_RADIUS:
                hit = 1
                nearest_t = t
                normal = ti.Vector([0.0, -1.0, 0.0])

    return hit, nearest_t, normal


@ti.func
def intersect_cone(
    ray_origin: ti.types.vector(3, ti.f32),
    ray_dir: ti.types.vector(3, ti.f32),
):
    """
    计算射线与完整圆锥的最近正交点。

    完整圆锥由两部分组成：
        1. 圆锥侧面
        2. 圆锥底面圆盘
    """
    hit = 0
    nearest_t = INF
    normal = ti.Vector([0.0, 0.0, 0.0])

    side_hit, side_t, side_normal = intersect_cone_side(ray_origin, ray_dir)
    base_hit, base_t, base_normal = intersect_cone_base(ray_origin, ray_dir)

    if side_hit == 1 and side_t < nearest_t:
        hit = 1
        nearest_t = side_t
        normal = side_normal

    if base_hit == 1 and base_t < nearest_t:
        hit = 1
        nearest_t = base_t
        normal = base_normal

    return hit, nearest_t, normal