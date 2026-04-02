# src/Work2/main.py
import taichi as ti

ti.init(arch=ti.gpu)

from .config import *
from .physics import *

def main():
    vertices[0] = [2.0, 0.0, -2.0]
    vertices[1] = [0.0, 2.0, -2.0]
    vertices[2] = [-2.0, 0.0, -2.0]

    gui = ti.GUI("3D Transformation (Taichi)", res=WINDOW_RES)

    angle = 0.0

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                angle += 10.0
            elif gui.event.key == 'd':
                angle -= 10.0
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False

        update_coordinates(angle)

        a = screen_coords[0]
        b = screen_coords[1]
        c = screen_coords[2]

        gui.line(a, b, radius=2, color=0xFF0000)
        gui.line(b, c, radius=2, color=0x00FF00)
        gui.line(c, a, radius=2, color=0x0000FF)

        gui.show()


if __name__ == "__main__":
    main()