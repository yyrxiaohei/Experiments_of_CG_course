# src/Work4/config.py

# 1. 窗口与画布配置
WINDOW_TITLE = "Experiment 4: Phong Lighting Model"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
WINDOW_RES = (WINDOW_WIDTH, WINDOW_HEIGHT)

# 2. 摄像机与成像平面配置
CAMERA_POS = (0.0, 0.0, 5.0)
SCREEN_Z = 0.0
VIEW_HALF_WIDTH = 3.4
VIEW_HALF_HEIGHT = 3.4

# 3. 点光源配置
LIGHT_POS = (2.0, 3.0, 4.0)
LIGHT_COLOR = (1.0, 1.0, 1.0)

# 4. 背景颜色配置
BACKGROUND_COLOR = (0.02, 0.12, 0.14)

# 5. 球体配置
SPHERE_CENTER = (-1.2, -0.2, 0.0)
SPHERE_RADIUS = 1.2
SPHERE_COLOR = (0.8, 0.1, 0.1)

# 6. 圆锥配置
CONE_APEX = (1.2, 1.2, 0.0)
CONE_BASE_Y = -1.4
CONE_BASE_RADIUS = 1.2
CONE_COLOR = (0.6, 0.2, 0.8)

# 7. Phong 光照参数默认值
DEFAULT_KA = 0.2
DEFAULT_KD = 0.7
DEFAULT_KS = 0.5
DEFAULT_SHININESS = 32.0

# 8. UI 滑动条参数范围
KA_MIN = 0.0
KA_MAX = 1.0

KD_MIN = 0.0
KD_MAX = 1.0

KS_MIN = 0.0
KS_MAX = 1.0

SHININESS_MIN = 1.0
SHININESS_MAX = 128.0

# 9. 求交与数值稳定配置
INF = 1.0e10
EPS = 1.0e-5
T_MIN = 1.0e-4

# 10. 选做功能开关
# False：使用标准 Phong 模型
# True ：使用 Blinn-Phong 半程向量模型
ENABLE_BLINN_PHONG = False

# False：不计算阴影
# True ：开启硬阴影，阴影区域只保留环境光
ENABLE_HARD_SHADOW = True