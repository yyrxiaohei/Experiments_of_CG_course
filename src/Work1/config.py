# src/Work0/config.py

# --- 物理系统参数 ---
NUM_PARTICLES = 10000      # 粒子总数 (卡顿请调小此数值，如 2000)
GRAVITY_STRENGTH = 0.001   # 鼠标引力强度
DRAG_COEF = 0.98           # 空气阻力系数
BOUNCE_COEF = -0.8         # 边界反弹能量损耗# --- 渲染系统参数 ---
WINDOW_RES = (800, 600)    # 窗口分辨率
PARTICLE_RADIUS = 1.5      # 粒子绘制半径
PARTICLE_COLOR = 0x00BFFF  # 粒子颜色 (天蓝色)