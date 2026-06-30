from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt


def save_progress_image(
    target_silhouette,
    pred_silhouette,
    epoch: int,
    epochs: int,
    loss_value: float,
    loss_silhouette_value: float,
    save_path: Path,
) -> None:
    """
    保存目标剪影和当前预测剪影的对比图。
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    fig.suptitle(
        f"迭代步数: {epoch:03d}/{epochs} | "
        f"总 Loss: {loss_value:.4f} | "
        f"剪影误差: {loss_silhouette_value:.4f}",
        fontsize=12,
    )

    ax[0].imshow(target_silhouette[0].detach().cpu().numpy(), cmap="gray")
    ax[0].set_title("Ground Truth Silhouette")
    ax[0].axis("off")

    ax[1].imshow(pred_silhouette[0].detach().cpu().numpy(), cmap="gray")
    ax[1].set_title(f"Optimizing... (Epoch {epoch})")
    ax[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_loss_curve(history, save_path: Path) -> None:
    """
    保存总损失及各项损失曲线。
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))

    plt.plot(history["loss"], label="Total Loss")
    plt.plot(history["loss_silhouette"], label="Silhouette Loss")
    plt.plot(history["loss_laplacian"], label="Laplacian Loss")
    plt.plot(history["loss_edge"], label="Edge Loss")
    plt.plot(history["loss_normal"], label="Normal Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Work6 Differentiable Rendering Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def make_optimization_gif(image_paths, save_path: Path) -> None:
    """
    将阶段性结果图合成为 GIF。
    """
    if not image_paths:
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [imageio.imread(path) for path in image_paths]
    imageio.mimsave(save_path, frames, duration=0.3)
