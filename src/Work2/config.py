# src/Work2/config.py
from re import ASCII

WINDOW_RES = (700, 700)

# 图形维度参数
VERTICE_NUM = 3
SPATIAL_DIMENSION = 3

# 相机位置参数
EYE_POS_X = 0.0
EYE_POS_Y = 0.0
EYE_POS_Z = 20.0

# 投影矩阵参数
EYE_FOV = 45.0  # 视野大小
ASPECT_RATIO = 1.0  # 屏幕宽高比
Z_NEAR = 0.1  # 最近可见距离
Z_FAR = 50.0  # 最远可见距离