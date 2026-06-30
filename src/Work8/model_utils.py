"""SMPL 模型路径解析、旧版 Chumpy 兼容和模型加载。"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import smplx

from .config import MODEL_EXTENSION, MODEL_GENDER, MODEL_TYPE


class _ChumpyArrayShim:
    """读取旧版 SMPL pickle 中 chumpy.Ch 数组所需的最小兼容类。"""

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _array(self):
        if hasattr(self, "r"):
            return self.r
        if hasattr(self, "x"):
            return self.x
        raise AttributeError("无法从旧版 chumpy pickle 对象中恢复数组数据")

    def __array__(self, dtype=None):
        return np.asarray(self._array(), dtype=dtype)

    @property
    def shape(self):
        return np.asarray(self).shape

    def __len__(self):
        return len(np.asarray(self))

    def __getitem__(self, item):
        return np.asarray(self)[item]


def install_chumpy_pickle_shim() -> None:
    """在未安装 Chumpy 时，为旧版 SMPL pickle 注册最小反序列化兼容层。"""

    if "chumpy.ch" in sys.modules:
        return

    chumpy_module = types.ModuleType("chumpy")
    chumpy_ch_module = types.ModuleType("chumpy.ch")

    _ChumpyArrayShim.__name__ = "Ch"
    _ChumpyArrayShim.__qualname__ = "Ch"
    _ChumpyArrayShim.__module__ = "chumpy.ch"

    chumpy_ch_module.Ch = _ChumpyArrayShim
    chumpy_module.ch = chumpy_ch_module

    sys.modules["chumpy"] = chumpy_module
    sys.modules["chumpy.ch"] = chumpy_ch_module


@dataclass(frozen=True)
class ModelInfo:
    """实验摘要中需要记录的 SMPL 基础信息。"""

    num_vertices: int
    num_faces: int
    num_joints: int
    num_betas: int


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    """将命令行中的相对路径解释为相对于 Work8 目录的路径。"""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def expected_model_candidates(model_path: Path) -> tuple[Path, ...]:
    """返回本实验允许的常见 SMPL_NEUTRAL.pkl 放置位置。"""

    if model_path.is_file():
        return (model_path,)
    return (
        model_path / "smpl" / "SMPL_NEUTRAL.pkl",
        model_path / "SMPL_NEUTRAL.pkl",
    )


def validate_model_path(model_path: Path) -> None:
    """在调用 smplx 前给出比底层 FileNotFoundError 更清楚的提示。"""

    if any(candidate.is_file() for candidate in expected_model_candidates(model_path)):
        return

    expected = "\n".join(f"  - {item}" for item in expected_model_candidates(model_path))
    raise FileNotFoundError(
        "未找到 SMPL 中性模型文件。请将 SMPL_NEUTRAL.pkl 放到以下任一位置：\n"
        f"{expected}\n"
        "模型文件受 SMPL 许可协议约束，不应直接提交到公开仓库。"
    )


def load_smpl_model(
    model_path: Path,
    num_betas: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
):
    """使用 ``smplx.create`` 加载 neutral SMPL 模型。"""

    validate_model_path(model_path)
    install_chumpy_pickle_shim()

    model = smplx.create(
        model_path=str(model_path),
        model_type=MODEL_TYPE,
        gender=MODEL_GENDER,
        ext=MODEL_EXTENSION,
        num_betas=num_betas,
        use_pca=False,
    )
    model = model.to(device=device, dtype=dtype)
    model.eval()

    if model.v_template.ndim != 2 or model.v_template.shape[1] != 3:
        raise RuntimeError(f"v_template 形状异常：{tuple(model.v_template.shape)}")
    if model.lbs_weights.ndim != 2:
        raise RuntimeError(f"lbs_weights 形状异常：{tuple(model.lbs_weights.shape)}")
    if model.shapedirs.shape[-1] < num_betas:
        raise ValueError(
            f"模型仅包含 {model.shapedirs.shape[-1]} 个 shape directions，"
            f"不能使用 num_betas={num_betas}"
        )

    return model


def collect_model_info(model, faces: np.ndarray, num_betas: int) -> ModelInfo:
    """收集任务 1 要求输出的顶点、面片、关节与 beta 维度。"""

    return ModelInfo(
        num_vertices=int(model.v_template.shape[0]),
        num_faces=int(faces.shape[0]),
        num_joints=int(model.lbs_weights.shape[1]),
        num_betas=int(num_betas),
    )


def choose_device(device_name: str) -> torch.device:
    """解析 ``cpu``、``cuda`` 或 ``auto`` 设备参数。"""

    normalized = device_name.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前 PyTorch 未检测到可用 CUDA 设备")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError(f"不支持的设备：{device_name}")
    return torch.device(normalized)
