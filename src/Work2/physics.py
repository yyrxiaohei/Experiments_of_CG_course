# src/Work2/physics.py
import taichi as ti
from .config import *

# 1 数据结构定义


# 2 内核定义

@ti.kernel
def get_model_matrix(angle):
    print("我是占位符")  # 记得删除

@ti.kernel
def get_view_matrix(eye_pos):
    print("我是占位符")  # 记得删除

@ti.kernel
def get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar):
    print("我是占位符")  # 记得删除
