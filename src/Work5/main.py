# src/Work5/main.py

import taichi as ti

# 注意：初始化必须在最前面执行，接管底层 GPU
ti.init(arch=ti.gpu)

from .config import *
from .raytracer import pixels, render


def main():
    print("正在编译 GPU 光线追踪内核，请稍候...")

    window = ti.ui.Window(
        "Experiment 5: Whitted-Style Ray Tracing",
        res=WINDOW_RES
    )

    canvas = window.get_canvas()
    gui = window.get_gui()

    light_x = DEFAULT_LIGHT_X
    light_y = DEFAULT_LIGHT_Y
    light_z = DEFAULT_LIGHT_Z
    max_bounces = DEFAULT_MAX_BOUNCES

    # 先渲染一帧，触发 Taichi Kernel 编译
    render(light_x, light_y, light_z, max_bounces)

    print("编译完成！")
    print("可通过右侧 UI 面板调节点光源位置与最大弹射次数。")

    while window.running:
        with gui.sub_window("Ray Tracing Parameters", 16, 16, 300, 220):
            light_x = gui.slider_float("Light X", light_x, -6.0, 6.0)
            light_y = gui.slider_float("Light Y", light_y, 0.5, 8.0)
            light_z = gui.slider_float("Light Z", light_z, -6.0, 6.0)

            max_bounces = gui.slider_int(
                "Max Bounces",
                max_bounces,
                MIN_MAX_BOUNCES,
                MAX_MAX_BOUNCES
            )

            gui.text("Max Bounces = 1: almost no mirror reflection")
            gui.text("Max Bounces > 1: reflected world appears")

        render(light_x, light_y, light_z, max_bounces)

        canvas.set_image(pixels)
        window.show()


if __name__ == "__main__":
    main()