# src/Work2/main.py
import taichi as ti

ti.init(arch=ti.gpu)  # 初始化，接管底层 GPU

# 自定义模块导入
from .config import *
from .physics import *

def main():
    # 初始化三角形顶点
    ''' 函数化、循环化 '''
    vertices[0] = [2.0, 0.0, -2.0]
    vertices[1] = [0.0, 2.0, -2.0]
    vertices[2] = [-2.0, 0.0, -2.0]

    # 创建 GUI 窗口
    gui = ti.GUI("3D Transformation (Taichi)", res=WINDOW_RES)

    while gui.running:
        # 通过Taichi内置固定常量，获取操作信息
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                angle += 10.0
            elif gui.event.key == 'd':
                angle -= 10.0
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False

        # 计算三维坐标变换
        update_coordinates(angle)
        
        # 获取二维投影坐标并绘制
        ''' 函数化、循环化 '''
        a = screen_coords[0]
        b = screen_coords[1]
        c = screen_coords[2]

        gui.line(a, b, radius=2, color=0xFF0000)
        gui.line(b, c, radius=2, color=0x00FF00)
        gui.line(c, a, radius=2, color=0x0000FF)
        
        gui.show()


if __name__ == "__main__":
    main()