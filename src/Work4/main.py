# src/Work4/main.py

from __future__ import annotations

import taichi as ti

# 注意：Taichi 初始化必须尽量靠前执行
ti.init(arch=ti.gpu)

from .config import (  # noqa: E402
    DEFAULT_KA,
    DEFAULT_KD,
    DEFAULT_KS,
    DEFAULT_SHININESS,
    KA_MAX,
    KA_MIN,
    KD_MAX,
    KD_MIN,
    KS_MAX,
    KS_MIN,
    SHININESS_MAX,
    SHININESS_MIN,
    WINDOW_RES,
    WINDOW_TITLE,
)
from .renderer import (  # noqa: E402
    init_material_params,
    pixels,
    render_scene,
    set_material_params,
)


def print_help_message():
    """
    打印实验四交互说明。
    """
    print("============================================================")
    print("Experiment 4: Phong Lighting Model")
    print("------------------------------------------------------------")
    print("实验内容：")
    print("1. 使用 Ray Casting 渲染红色球体与紫色圆锥。")
    print("2. 实现球体、圆锥求交与最近深度竞争。")
    print("3. 实现 Ambient、Diffuse、Specular 三项 Phong 光照。")
    print("4. 通过 UI Slider 实时调节 Ka、Kd、Ks、Shininess。")
    print("------------------------------------------------------------")
    print("交互说明：")
    print("ESC：退出程序")
    print("左侧 UI 面板：实时调节 Phong 光照参数")
    print("============================================================")


def main():
    print("正在初始化 GPU Field 与 GGUI 窗口，请稍候...")

    init_material_params()

    ka = DEFAULT_KA
    kd = DEFAULT_KD
    ks = DEFAULT_KS
    shininess = DEFAULT_SHININESS

    window = ti.ui.Window(
        name=WINDOW_TITLE,
        res=WINDOW_RES,
        vsync=True,
    )

    canvas = window.get_canvas()
    gui = window.get_gui()

    print_help_message()
    print("初始化完成！请在窗口左侧调节 Phong 光照参数。")

    while window.running:
        # =========================
        # 事件处理
        # =========================
        while window.get_event(ti.ui.PRESS):
            event_key = window.event.key

            if event_key == ti.ui.ESCAPE:
                window.running = False

        # =========================
        # UI 参数面板
        # =========================
        with gui.sub_window("Phong Parameters", 0.02, 0.02, 0.32, 0.28):
            gui.text("Local Illumination Model")

            ka = gui.slider_float("Ka Ambient", ka, KA_MIN, KA_MAX)
            kd = gui.slider_float("Kd Diffuse", kd, KD_MIN, KD_MAX)
            ks = gui.slider_float("Ks Specular", ks, KS_MIN, KS_MAX)
            shininess = gui.slider_float(
                "Shininess",
                shininess,
                SHININESS_MIN,
                SHININESS_MAX,
            )

            gui.text(f"Ka = {ka:.2f}")
            gui.text(f"Kd = {kd:.2f}")
            gui.text(f"Ks = {ks:.2f}")
            gui.text(f"Shininess = {shininess:.1f}")

        # =========================
        # 更新材质参数
        # =========================
        set_material_params(
            ka=ka,
            kd=kd,
            ks=ks,
            shininess=shininess,
        )

        # =========================
        # GPU 光线投射渲染
        # =========================
        render_scene()

        # =========================
        # GGUI 显示
        # =========================
        canvas.set_image(pixels)
        window.show()


if __name__ == "__main__":
    main()