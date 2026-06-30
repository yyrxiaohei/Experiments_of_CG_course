from pathlib import Path

import torch
from pytorch3d.loss import (
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

from .config import (
    DISPLAY_INTERVAL,
    EDGE_WEIGHT,
    EPOCHS,
    LAPLACIAN_WEIGHT,
    LEARNING_RATE,
    MOMENTUM,
    NORMAL_WEIGHT,
    SAVE_PROGRESS_IMAGES,
)
from .mesh_utils import save_mesh
from .renderer import render_predicted_silhouette
from .visualization import save_progress_image


def try_clear_output() -> None:
    """
    在 Notebook 环境中清空旧输出；普通终端环境下静默跳过。
    """
    try:
        from IPython.display import clear_output

        clear_output(wait=True)
    except Exception:
        pass


def optimize_mesh(
    src_mesh,
    target_silhouette,
    rasterizer,
    shader,
    image_dir: Path,
    mesh_dir: Path,
):
    """
    使用可微软剪影渲染优化球体顶点，使其逼近目标奶牛模型。

    核心实现尽量保持老师参考代码：
    - deform_verts 为可学习顶点偏移；
    - 使用 SGD + momentum；
    - 使用 offset_verts 构造当前网格；
    - 采用剪影 MSE 与三种网格正则项。
    """
    deform_verts = torch.zeros_like(
        src_mesh.verts_packed(),
        requires_grad=True,
    )

    optimizer = torch.optim.SGD(
        [deform_verts],
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
    )

    history = {
        "loss": [],
        "loss_silhouette": [],
        "loss_laplacian": [],
        "loss_edge": [],
        "loss_normal": [],
        "saved_images": [],
    }

    for i in range(EPOCHS):
        optimizer.zero_grad()

        new_src_mesh = src_mesh.offset_verts(deform_verts)

        pred_silhouette = render_predicted_silhouette(
            src_mesh=new_src_mesh,
            rasterizer=rasterizer,
            shader=shader,
        )

        loss_silhouette = ((pred_silhouette - target_silhouette) ** 2).mean()
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh)
        loss_edge = mesh_edge_loss(new_src_mesh)
        loss_normal = mesh_normal_consistency(new_src_mesh)

        loss = (
            loss_silhouette
            + LAPLACIAN_WEIGHT * loss_laplacian
            + EDGE_WEIGHT * loss_edge
            + NORMAL_WEIGHT * loss_normal
        )

        loss.backward()
        optimizer.step()

        history["loss"].append(float(loss.detach().cpu()))
        history["loss_silhouette"].append(
            float(loss_silhouette.detach().cpu())
        )
        history["loss_laplacian"].append(
            float(loss_laplacian.detach().cpu())
        )
        history["loss_edge"].append(float(loss_edge.detach().cpu()))
        history["loss_normal"].append(float(loss_normal.detach().cpu()))

        if i % DISPLAY_INTERVAL == 0 or i == EPOCHS - 1:
            try_clear_output()

            print(
                f"迭代步数: {i:03d}/{EPOCHS} | "
                f"总 Loss: {loss.item():.4f} | "
                f"剪影误差: {loss_silhouette.item():.4f}"
            )

            mesh_path = mesh_dir / f"mesh_epoch_{i:03d}.obj"
            save_mesh(new_src_mesh, mesh_path)
            print(f"[*] 已保存当前 3D 模型至: {mesh_path}")

            if SAVE_PROGRESS_IMAGES:
                image_path = image_dir / f"epoch_{i:03d}.png"
                save_progress_image(
                    target_silhouette=target_silhouette,
                    pred_silhouette=pred_silhouette,
                    epoch=i,
                    epochs=EPOCHS,
                    loss_value=float(loss.detach().cpu()),
                    loss_silhouette_value=float(
                        loss_silhouette.detach().cpu()
                    ),
                    save_path=image_path,
                )
                history["saved_images"].append(image_path)
                print(f"[*] 已保存当前剪影对比图至: {image_path}")

    # 使用最后一次 optimizer.step() 更新后的 deform_verts 重新构造最终网格。
    final_mesh = src_mesh.offset_verts(deform_verts.detach())
    return final_mesh, history
