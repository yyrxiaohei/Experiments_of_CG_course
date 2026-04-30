# src/Work5/raytracer.py

import taichi as ti
import math

from .config import *


# ====== 1 图像缓冲区定义 ======
pixels = ti.Vector.field(3, dtype=ti.f32, shape=WINDOW_RES)


# ====== 2 基础工具函数 ======
@ti.func
def clamp_color(color):
    """
    将颜色限制在 [0, 1] 范围内，避免过曝或非法颜色值。
    """
    return ti.Vector([
        ti.min(1.0, ti.max(0.0, color[0])),
        ti.min(1.0, ti.max(0.0, color[1])),
        ti.min(1.0, ti.max(0.0, color[2])),
    ])


@ti.func
def reflect(in_dir, normal):
    """
    根据反射公式计算理想镜面反射方向：
    R = L_in - 2 * dot(L_in, N) * N
    """
    return in_dir - 2.0 * in_dir.dot(normal) * normal


@ti.func
def get_background_color(ray_dir):
    """
    简单天空背景：根据光线 y 方向做渐变。
    """
    t = 0.5 * (ray_dir[1] + 1.0)

    bottom_color = ti.Vector([0.75, 0.82, 0.92])
    top_color = ti.Vector([0.25, 0.45, 0.85])

    return (1.0 - t) * bottom_color + t * top_color


# ====== 3 材质颜色函数 ======
@ti.func
def get_base_color(mat_id, hit_pos):
    """
    根据材质 ID 和交点位置返回基础颜色。

    地面使用黑白棋盘格纹理：
    通过交点的 x、z 坐标所在网格奇偶性判断颜色。
    """
    color = ti.Vector([1.0, 1.0, 1.0])

    if mat_id == MAT_GROUND:
        checker = ti.cast(ti.floor(hit_pos[0]) + ti.floor(hit_pos[2]), ti.i32) % 2

        if checker == 0:
            color = ti.Vector([0.88, 0.88, 0.88])
        else:
            color = ti.Vector([0.12, 0.12, 0.12])

    elif mat_id == MAT_RED_DIFFUSE:
        color = ti.Vector([0.95, 0.12, 0.08])

    elif mat_id == MAT_SILVER_MIRROR:
        color = ti.Vector([0.85, 0.85, 0.82])

    return color


# ====== 4 几何求交函数 ======
@ti.func
def intersect_sphere(ray_origin, ray_dir, center, radius):
    """
    光线与球体求交。

    返回：
    hit: 是否相交
    t: 最近交点的参数距离
    normal: 交点法向量
    """
    hit = False
    t = INF
    normal = ti.Vector([0.0, 1.0, 0.0])

    oc = ray_origin - center

    a = ray_dir.dot(ray_dir)
    b = 2.0 * oc.dot(ray_dir)
    c = oc.dot(oc) - radius * radius

    discriminant = b * b - 4.0 * a * c

    if discriminant > 0.0:
        sqrt_d = ti.sqrt(discriminant)

        t1 = (-b - sqrt_d) / (2.0 * a)
        t2 = (-b + sqrt_d) / (2.0 * a)

        if t1 > RAY_EPSILON:
            hit = True
            t = t1
        elif t2 > RAY_EPSILON:
            hit = True
            t = t2

        if hit:
            hit_pos = ray_origin + t * ray_dir
            normal = (hit_pos - center).normalized()

    return hit, t, normal


@ti.func
def intersect_ground(ray_origin, ray_dir):
    """
    光线与无限大水平地面 y = GROUND_Y 求交。
    """
    hit = False
    t = INF
    normal = ti.Vector([0.0, 1.0, 0.0])

    if ti.abs(ray_dir[1]) > 1e-6:
        temp_t = (GROUND_Y - ray_origin[1]) / ray_dir[1]

        if temp_t > RAY_EPSILON:
            hit = True
            t = temp_t

    return hit, t, normal


@ti.func
def trace_scene(ray_origin, ray_dir):
    """
    对当前场景中所有隐式几何体做求交，返回最近交点信息。

    场景包含：
    1. 无限大棋盘格地面
    2. 左侧红色漫反射球
    3. 右侧银色镜面球
    """
    hit_anything = False
    closest_t = INF

    hit_pos = ti.Vector([0.0, 0.0, 0.0])
    hit_normal = ti.Vector([0.0, 1.0, 0.0])
    hit_mat_id = MAT_NONE

    # 地面求交
    ground_hit, ground_t, ground_normal = intersect_ground(ray_origin, ray_dir)

    if ground_hit and ground_t < closest_t:
        hit_anything = True
        closest_t = ground_t
        hit_pos = ray_origin + ground_t * ray_dir
        hit_normal = ground_normal
        hit_mat_id = MAT_GROUND

    # 红色漫反射球求交
    red_center = ti.Vector([RED_SPHERE_X, RED_SPHERE_Y, RED_SPHERE_Z])
    red_hit, red_t, red_normal = intersect_sphere(
        ray_origin,
        ray_dir,
        red_center,
        RED_SPHERE_RADIUS
    )

    if red_hit and red_t < closest_t:
        hit_anything = True
        closest_t = red_t
        hit_pos = ray_origin + red_t * ray_dir
        hit_normal = red_normal
        hit_mat_id = MAT_RED_DIFFUSE

    # 银色镜面球求交
    mirror_center = ti.Vector([MIRROR_SPHERE_X, MIRROR_SPHERE_Y, MIRROR_SPHERE_Z])
    mirror_hit, mirror_t, mirror_normal = intersect_sphere(
        ray_origin,
        ray_dir,
        mirror_center,
        MIRROR_SPHERE_RADIUS
    )

    if mirror_hit and mirror_t < closest_t:
        hit_anything = True
        closest_t = mirror_t
        hit_pos = ray_origin + mirror_t * ray_dir
        hit_normal = mirror_normal
        hit_mat_id = MAT_SILVER_MIRROR

    return hit_anything, closest_t, hit_pos, hit_normal, hit_mat_id


# ====== 5 阴影与着色函数 ======
@ti.func
def is_in_shadow(hit_pos, hit_normal, light_pos):
    """
    硬阴影检测。

    从交点向点光源发射暗影射线：
    若暗影射线在到达光源之前击中其他物体，则该点处于阴影中。
    """
    shadow = False

    shadow_origin = hit_pos + hit_normal * RAY_EPSILON
    to_light = light_pos - shadow_origin

    light_distance = to_light.norm()
    shadow_dir = to_light.normalized()

    shadow_hit, shadow_t, shadow_pos, shadow_normal, shadow_mat_id = trace_scene(
        shadow_origin,
        shadow_dir
    )

    if shadow_hit and shadow_t < light_distance:
        shadow = True

    return shadow


@ti.func
def shade_diffuse(hit_pos, hit_normal, mat_id, ray_dir, light_pos):
    """
    对漫反射材质执行 Phong 光照计算。

    包含：
    1. 环境光
    2. 漫反射
    3. 高光项
    4. 硬阴影
    """
    base_color = get_base_color(mat_id, hit_pos)

    ambient = AMBIENT_STRENGTH * base_color
    final_color = ambient

    shadow = is_in_shadow(hit_pos, hit_normal, light_pos)

    if not shadow:
        light_dir = (light_pos - hit_pos).normalized()
        view_dir = (-ray_dir).normalized()

        # 漫反射项
        diff = ti.max(hit_normal.dot(light_dir), 0.0)
        diffuse = DIFFUSE_STRENGTH * diff * base_color

        # Phong 高光项
        reflect_dir = reflect(-light_dir, hit_normal).normalized()
        spec = ti.pow(ti.max(view_dir.dot(reflect_dir), 0.0), SHININESS)
        specular = SPECULAR_STRENGTH * spec * ti.Vector([1.0, 1.0, 1.0])

        final_color += diffuse + specular

    return final_color


# ====== 6 相机光线生成函数 ======
@ti.func
def generate_camera_ray(i, j):
    """
    为像素 (i, j) 生成一条主光线。

    这里使用简单针孔相机模型：
    - 相机位于 CAMERA_POS
    - 看向 CAMERA_LOOK_AT
    - 根据 FOV 和屏幕宽高比生成成像平面方向
    """
    camera_pos = ti.Vector([CAMERA_POS_X, CAMERA_POS_Y, CAMERA_POS_Z])
    look_at = ti.Vector([CAMERA_LOOK_AT_X, CAMERA_LOOK_AT_Y, CAMERA_LOOK_AT_Z])
    world_up = ti.Vector([CAMERA_UP_X, CAMERA_UP_Y, CAMERA_UP_Z])

    forward = (look_at - camera_pos).normalized()
    right = forward.cross(world_up).normalized()
    up = right.cross(forward).normalized()

    aspect_ratio = IMAGE_WIDTH / IMAGE_HEIGHT
    fov_rad = CAMERA_FOV * math.pi / 180.0
    scale = ti.tan(fov_rad * 0.5)

    x = (2.0 * (ti.cast(i, ti.f32) + 0.5) / IMAGE_WIDTH - 1.0) * aspect_ratio * scale
    y = (2.0 * (ti.cast(j, ti.f32) + 0.5) / IMAGE_HEIGHT - 1.0) * scale

    ray_dir = (forward + x * right + y * up).normalized()

    return camera_pos, ray_dir


# ====== 7 核心渲染 Kernel ======
@ti.kernel
def render(light_x: ti.f32, light_y: ti.f32, light_z: ti.f32, max_bounces: ti.i32):
    """
    基于迭代循环的 Whitted-Style 光线追踪。

    每个像素独立执行：
    1. 从相机发出 Primary Ray
    2. 若击中漫反射材质，执行直接光照与阴影检测，然后终止
    3. 若击中镜面材质，生成反射射线，继续循环
    4. 达到最大弹射次数后停止
    """
    light_pos = ti.Vector([light_x, light_y, light_z])

    for i, j in pixels:
        ray_origin, ray_dir = generate_camera_ray(i, j)

        throughput = ti.Vector([1.0, 1.0, 1.0])
        final_color = ti.Vector([0.0, 0.0, 0.0])

        finished = False

        for bounce in range(MAX_MAX_BOUNCES):
            if not finished:
                if bounce < max_bounces:
                    hit, hit_t, hit_pos, hit_normal, mat_id = trace_scene(ray_origin, ray_dir)

                    if hit:
                        if mat_id == MAT_SILVER_MIRROR:
                            # 镜面材质：计算反射方向，继续追踪反射射线
                            ray_origin = hit_pos + hit_normal * RAY_EPSILON
                            ray_dir = reflect(ray_dir, hit_normal).normalized()

                            throughput *= MIRROR_REFLECTANCE

                        else:
                            # 漫反射材质：计算局部光照并终止当前路径
                            local_color = shade_diffuse(
                                hit_pos,
                                hit_normal,
                                mat_id,
                                ray_dir,
                                light_pos
                            )

                            final_color += throughput * local_color
                            finished = True

                    else:
                        # 未击中任何物体：返回天空背景
                        final_color += throughput * get_background_color(ray_dir)
                        finished = True

        pixels[i, j] = clamp_color(final_color)