# src/Work7/renderer.py

import taichi as ti

from .config import *
from .physics import positions, spring_indices


# ====== 1 渲染环境初始化 ======
def create_render_context():
    """创建 GGUI 窗口、画布、场景和相机。"""
    window = ti.ui.Window(
        WINDOW_TITLE,
        res=WINDOW_RES,
    )

    canvas = window.get_canvas()
    scene = window.get_scene()

    camera = ti.ui.Camera()
    camera.position(*CAMERA_POSITION)
    camera.lookat(*CAMERA_LOOK_AT)
    camera.up(*CAMERA_UP)
    camera.fov(CAMERA_FOV)

    return window, canvas, scene, camera


# ====== 2 三维场景渲染 ======
def render_scene(window, canvas, scene, camera):
    """更新相机并绘制质点和结构弹簧。"""
    camera.track_user_inputs(
        window,
        movement_speed=CAMERA_MOVEMENT_SPEED,
        hold_key=ti.ui.RMB,
    )

    scene.set_camera(camera)

    scene.ambient_light(AMBIENT_LIGHT_COLOR)
    scene.point_light(
        pos=POINT_LIGHT_POSITION,
        color=POINT_LIGHT_COLOR,
    )

    scene.particles(
        positions,
        radius=PARTICLE_RADIUS,
        color=PARTICLE_COLOR,
    )

    scene.lines(
        positions,
        indices=spring_indices,
        width=SPRING_LINE_WIDTH,
        color=SPRING_LINE_COLOR,
    )

    canvas.scene(scene)
