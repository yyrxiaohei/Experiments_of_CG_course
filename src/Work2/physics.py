# src/Work2/physics.py
import taichi as ti
import math
from .config import *

# 1 数据定义
vertices = ti.Vector.field(SPATIAL_DIMENSION, dtype=ti.f32, shape=VERTICE_NUM)  # 三维坐标
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=VERTICE_NUM)  # 屏幕投影坐标
angle = 0.0  # 当前旋转角

# 2 三大矩阵变换函数定义

@ti.func
def get_model_matrix(angle):
    '''
    接收一个旋转角度（角度制），返回绕 Z 轴旋转该角度的模型变换矩阵
    '''
    print("我是占位符")  # 记得删除

@ti.func
def get_view_matrix(eye_pos):
    '''
    接收相机位置（三维向量），返回视图变换矩阵
    其中，将相机平移至世界坐标系的原点
    '''
    print("我是占位符")  # 记得删除

@ti.func
def get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar):
    '''
    接收视场角（Y 轴方向，角度制）、屏幕长宽比、近截面距离和远截面距离，返回透视投影矩阵
    '''
    print("我是占位符")  # 记得删除


# 3 核心内核定义

@ti.kernel
def update_coordinates(angle: ti.f32):
    '''
    坐标变换并行计算
    '''
    # 相机位置设置
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    
    # 计算 MVP 矩阵（模型变换 -> 观察变换 -> 投影变换）
    model = get_model_matrix(angle)  # M 矩阵，模型变换
    view = get_view_matrix(eye_pos)  # V 矩阵，观察变换
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)  # P 矩阵，投影变换
    mvp = proj @ view @ model  # 合成 MVP 矩阵

    # 对图形的每个顶点独立执行 MVP 变换
    for i in range(VERTICE_NUM):
        # 标记单顶点
        v = vertices[i]

        # 补全齐次坐标并执行 MVP 变换
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        
        # 透视除法，转化为 NDC 坐标 [-1, 1]
        v_ndc = v_clip / v_clip[3]

        # 视口变换：映射到 GUI 的 [0, 1] x [0, 1] 空间
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0







# @ti.func
# def get_model_matrix(angle: ti.f32):
#     """
#     模型变换矩阵：绕 Z 轴旋转
#     """
#     rad = angle * math.pi / 180.0
#     c = ti.cos(rad)
#     s = ti.sin(rad)
#     return ti.Matrix([
#         [c, -s, 0.0, 0.0],
#         [s,  c, 0.0, 0.0],
#         [0.0, 0.0, 1.0, 0.0],
#         [0.0, 0.0, 0.0, 1.0]
#     ])

# @ti.func
# def get_view_matrix(eye_pos):
#     """
#     视图变换矩阵：将相机移动到原点
#     """
#     return ti.Matrix([
#         [1.0, 0.0, 0.0, -eye_pos[0]],
#         [0.0, 1.0, 0.0, -eye_pos[1]],
#         [0.0, 0.0, 1.0, -eye_pos[2]],
#         [0.0, 0.0, 0.0, 1.0]
#     ])

# @ti.func
# def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32):
#     """
#     透视投影矩阵
#     """
#     # 视线看向 -Z 轴，实际坐标为负
#     n = -zNear
#     f = -zFar
    
#     # 视角转化为弧度并求出 t, b, r, l
#     fov_rad = eye_fov * math.pi / 180.0
#     t = ti.tan(fov_rad / 2.0) * ti.abs(n)
#     b = -t
#     r = aspect_ratio * t
#     l = -r
    
#     # 1. 挤压矩阵: 透视平截头体 -> 长方体
#     M_p2o = ti.Matrix([
#         [n, 0.0, 0.0, 0.0],
#         [0.0, n, 0.0, 0.0],
#         [0.0, 0.0, n + f, -n * f],
#         [0.0, 0.0, 1.0, 0.0]
#     ])
    
#     # 2. 正交投影矩阵: 缩放与平移至 [-1, 1]^3
#     M_ortho_scale = ti.Matrix([
#         [2.0 / (r - l), 0.0, 0.0, 0.0],
#         [0.0, 2.0 / (t - b), 0.0, 0.0],
#         [0.0, 0.0, 2.0 / (n - f), 0.0],
#         [0.0, 0.0, 0.0, 1.0]
#     ])
    
#     M_ortho_trans = ti.Matrix([
#         [1.0, 0.0, 0.0, -(r + l) / 2.0],
#         [0.0, 1.0, 0.0, -(t + b) / 2.0],
#         [0.0, 0.0, 1.0, -(n + f) / 2.0],
#         [0.0, 0.0, 0.0, 1.0]
#     ])
    
#     M_ortho = M_ortho_scale @ M_ortho_trans
    
#     # 返回组合矩阵
#     return M_ortho @ M_p2o

