# src/Work3/config.py

# 1 窗口与显存配置

WINDOW_TITLE = "Experiment 3: Bezier Curve Rasterization"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
WINDOW_RES = (WINDOW_WIDTH, WINDOW_HEIGHT)

# 2 曲线与控制点配置

NUM_SEGMENTS = 1000
MAX_CURVE_POINTS = NUM_SEGMENTS + 1

MAX_CONTROL_POINTS = 100
MIN_BEZIER_CONTROL_POINTS = 2
MIN_BSPLINE_CONTROL_POINTS = 4

# 3 渲染颜色配置

BACKGROUND_COLOR = (0.02, 0.02, 0.025)

CURVE_COLOR = (0.0, 1.0, 0.25)
CONTROL_POINT_COLOR = (1.0, 0.15, 0.15)
CONTROL_POLYGON_COLOR = (0.55, 0.55, 0.55)

# 4 渲染样式配置

CONTROL_POINT_RADIUS = 0.008
CONTROL_POLYGON_WIDTH = 0.002

# 5 反走样开关：

# False：基础光栅化，只点亮单个像素；
# True ：选做增强，点亮 3x3 邻域并按距离衰减混合颜色。
ENABLE_ANTI_ALIASING = False

# 反走样影响半径，单位为像素
ANTI_ALIASING_RADIUS = 1.5

# 6 对象池配置

# canvas.lines() 使用一组线段顶点：
# 0-1 为第一条线，2-3 为第二条线，以此类推。
MAX_CONTROL_LINE_VERTICES = 2 * (MAX_CONTROL_POINTS - 1)

# 屏幕外隐藏坐标，用于对象池占位
HIDDEN_POS = -10.0


# 7 曲线模式

MODE_BEZIER = 0
MODE_BSPLINE = 1