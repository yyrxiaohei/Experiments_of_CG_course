import os
from pathlib import Path

import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

from .config import DEVICE, SOURCE_SPHERE_LEVEL, TARGET_MESH_PATH


def get_device() -> torch.device:
    """
    获取运行设备。

    优先使用 config.py 中指定的 CUDA 设备；若 CUDA 不可用，
    则自动退回 CPU。
    """
    if "cuda" in DEVICE and torch.cuda.is_available():
        return torch.device(DEVICE)
    return torch.device("cpu")


def find_cow_obj_path() -> Path:
    """
    查找 cow.obj 文件。

    查找顺序：
    1. assets/Work6/cow.obj；
    2. 当前运行目录下的 cow.obj。
    """
    if TARGET_MESH_PATH.exists():
        return TARGET_MESH_PATH

    local_path = Path(os.path.abspath("cow.obj"))
    if local_path.exists():
        return local_path

    raise FileNotFoundError(
        "未找到 cow.obj。\n"
        f"请将模型文件放到以下任一位置：\n"
        f"1. {TARGET_MESH_PATH}\n"
        "2. 当前运行目录下的 cow.obj"
    )


def load_target_mesh(device: torch.device) -> Meshes:
    """
    读取并归一化目标奶牛模型。

    归一化方式与老师参考实现保持一致：
    先平移到原点附近，再用各坐标轴绝对值最大值中的最大者缩放。
    """
    obj_path = find_cow_obj_path()

    verts, faces, _ = load_obj(str(obj_path))
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)

    verts = verts - verts.mean(0)
    scale = verts.abs().max(0)[0].max()
    if scale <= 0:
        raise ValueError("cow.obj 顶点坐标无有效尺度，无法进行归一化。")
    verts = verts / scale

    return Meshes(verts=[verts], faces=[faces_idx])


def create_source_mesh(device: torch.device) -> Meshes:
    """
    创建作为优化起点的四级细分二十面体球。
    """
    return ico_sphere(SOURCE_SPHERE_LEVEL, device)


def save_mesh(mesh: Meshes, save_path: Path) -> None:
    """
    将 batch 中唯一一个网格保存为 OBJ。
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    current_verts = mesh.verts_list()[0].detach().cpu()
    current_faces = mesh.faces_list()[0].detach().cpu()

    save_obj(
        f=str(save_path),
        verts=current_verts,
        faces=current_faces,
    )
