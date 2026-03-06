# src/Work0/main.py
import taichi as ti

# 注意：初始化必须在最前面执行，接管底层 GPU
ti.init(arch=ti.gpu)

# 导入我们自己写的模块
from .config import WINDOW_RES, PARTICLE_COLOR, PARTICLE_RADIUS
from .physics import init_particles, update_particles, pos

def run():
    print("正在编译 GPU 内核，请稍候...")
    init_particles()
    
    gui = ti.GUI("Experiment 0: Taichi Gravity Swarm", res=WINDOW_RES)
    print("编译完成！请在弹出的窗口中移动鼠标。")
    
    # 渲染主循环
    while gui.running:
        mouse_x, mouse_y = gui.get_cursor_pos()
        
        # 驱动 GPU 进行物理计算
        update_particles(mouse_x, mouse_y)
        
        # 读取显存数据并绘制
        gui.circles(pos.to_numpy(), color=PARTICLE_COLOR, radius=PARTICLE_RADIUS)
        gui.show()

if __name__ == "__main__":
    run()