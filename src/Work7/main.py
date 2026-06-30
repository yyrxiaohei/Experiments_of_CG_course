# src/Work7/main.py

import taichi as ti

# 必须先初始化 Taichi，再导入创建数据场的物理模块。
ti.init(arch=ti.gpu)

from .config import *
from .physics import (
    init_cloth,
    step_explicit,
    step_implicit_iter,
    step_semi_implicit,
)
from .renderer import create_render_context, render_scene


# ====== 1 数值积分方法名称 ======
METHOD_NAMES = {
    METHOD_EXPLICIT_EULER: "Explicit Euler",
    METHOD_SEMI_IMPLICIT_EULER: "Semi-Implicit Euler",
    METHOD_IMPLICIT_EULER: "Implicit Euler (Fixed-Point)",
}


# ====== 2 单个物理子步调度 ======
def step_simulation(current_method):
    """根据当前选择调用对应积分 Kernel。"""
    if current_method == METHOD_EXPLICIT_EULER:
        step_explicit()
    elif current_method == METHOD_SEMI_IMPLICIT_EULER:
        step_semi_implicit()
    else:
        step_implicit_iter()


# ====== 3 主程序 ======
def main():
    init_cloth()

    window, canvas, scene, camera = create_render_context()
    gui = window.get_gui()

    current_method = DEFAULT_INTEGRATION_METHOD
    paused = False

    while window.running:
        # ---------- GGUI 控制面板 ----------
        with gui.sub_window(
            "Mass-Spring Control",
            GUI_PANEL_X,
            GUI_PANEL_Y,
            GUI_PANEL_WIDTH,
            GUI_PANEL_HEIGHT,
        ):
            gui.text("Integration Method:")
            gui.text(f"Current: {METHOD_NAMES[current_method]}")

            explicit_prefix = (
                "[*] " if current_method == METHOD_EXPLICIT_EULER else "[ ] "
            )
            semi_prefix = (
                "[*] "
                if current_method == METHOD_SEMI_IMPLICIT_EULER
                else "[ ] "
            )
            implicit_prefix = (
                "[*] " if current_method == METHOD_IMPLICIT_EULER else "[ ] "
            )

            if gui.button(explicit_prefix + "Explicit Euler (Explosive)"):
                current_method = METHOD_EXPLICIT_EULER
                init_cloth()

            if gui.button(semi_prefix + "Semi-Implicit Euler (Stable)"):
                current_method = METHOD_SEMI_IMPLICIT_EULER
                init_cloth()

            if gui.button(implicit_prefix + "Implicit Euler (Damped)"):
                current_method = METHOD_IMPLICIT_EULER
                init_cloth()

            gui.text("")

            pause_label = "Resume Simulation" if paused else "Pause Simulation"

            if gui.button(pause_label):
                paused = not paused

            if gui.button("Reset Cloth"):
                init_cloth()

            gui.text("")
            gui.text(f"Time Step: {TIME_STEP:.4g}")
            gui.text(f"Substeps / Frame: {SUBSTEPS_PER_FRAME}")
            gui.text(f"Stiffness: {STRUCTURAL_STIFFNESS:.1f}")
            gui.text(f"Damping: {DAMPING_COEFFICIENT:.1f}")

        # ---------- 物理模拟 ----------
        if not paused:
            for _ in range(SUBSTEPS_PER_FRAME):
                step_simulation(current_method)

        # ---------- 三维渲染 ----------
        render_scene(
            window,
            canvas,
            scene,
            camera,
        )

        window.show()


if __name__ == "__main__":
    main()
