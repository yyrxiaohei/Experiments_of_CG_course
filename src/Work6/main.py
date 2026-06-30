import random

import numpy as np
import pytorch3d
import torch

from .config import (
    GIF_NAME,
    IMAGE_DIR,
    MAKE_GIF,
    MESH_DIR,
    RANDOM_SEED,
    SAVE_DIR,
    SAVE_LOSS_CURVE,
)
from .mesh_utils import (
    create_source_mesh,
    get_device,
    load_target_mesh,
    save_mesh,
)
from .optimizer import optimize_mesh
from .renderer import (
    create_rasterizer_and_shader,
    render_target_silhouette,
)
from .visualization import make_optimization_gif, save_loss_curve


def set_random_seed(seed: int) -> None:
    """
    固定随机种子，提升实验复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dirs() -> None:
    """
    创建实验输出目录。
    """
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    MESH_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """
    实验六主入口。
    """
    set_random_seed(RANDOM_SEED)
    prepare_dirs()

    device = get_device()

    print(f"当前运行设备: {device}")
    print(f"PyTorch3D 版本: {pytorch3d.__version__}")

    print("正在加载并归一化目标奶牛模型...")
    cow_mesh = load_target_mesh(device)

    print("正在创建多视角软剪影渲染管线...")
    rasterizer, shader = create_rasterizer_and_shader(device)

    print("正在渲染目标奶牛的多视角剪影...")
    target_silhouette = render_target_silhouette(
        cow_mesh=cow_mesh,
        rasterizer=rasterizer,
        shader=shader,
    )

    print("正在创建初始球体网格...")
    src_mesh = create_source_mesh(device)

    save_mesh(cow_mesh, MESH_DIR / "target_cow_normalized.obj")
    save_mesh(src_mesh, MESH_DIR / "source_sphere.obj")

    print(f"中间模型将保存在目录: {MESH_DIR}")
    print("开始执行可微渲染优化...")

    final_mesh, history = optimize_mesh(
        src_mesh=src_mesh,
        target_silhouette=target_silhouette,
        rasterizer=rasterizer,
        shader=shader,
        image_dir=IMAGE_DIR,
        mesh_dir=MESH_DIR,
    )

    final_mesh_path = MESH_DIR / "final_optimized_mesh.obj"
    save_mesh(final_mesh, final_mesh_path)

    if SAVE_LOSS_CURVE:
        loss_curve_path = SAVE_DIR / "loss_curve.png"
        save_loss_curve(history, loss_curve_path)
        print(f"Loss 曲线已保存至: {loss_curve_path}")

    if MAKE_GIF:
        gif_path = SAVE_DIR / GIF_NAME
        make_optimization_gif(history["saved_images"], gif_path)
        print(f"优化过程 GIF 已保存至: {gif_path}")

    print("优化完成。")
    print(f"最终优化模型已保存至: {final_mesh_path}")
    print(f"阶段性剪影对比图已保存至: {IMAGE_DIR}")


if __name__ == "__main__":
    main()
