import numpy as np
import torch
from pytorch3d.renderer import (
    BlendParams,
    FoVPerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings,
    SoftSilhouetteShader,
    look_at_view_transform,
)

from .config import (
    CAMERA_AZIM_END,
    CAMERA_AZIM_START,
    CAMERA_DISTANCE,
    CAMERA_ELEVATION,
    FACES_PER_PIXEL,
    GAMMA,
    IMAGE_SIZE,
    NUM_VIEWS,
    SIGMA,
)


def create_cameras(device: torch.device) -> FoVPerspectiveCameras:
    """
    创建环绕目标模型分布的多视角透视相机。
    """
    elev = torch.full(
        (NUM_VIEWS,),
        fill_value=CAMERA_ELEVATION,
        dtype=torch.float32,
        device=device,
    )
    azim = torch.linspace(
        CAMERA_AZIM_START,
        CAMERA_AZIM_END,
        NUM_VIEWS,
        device=device,
    )

    R, T = look_at_view_transform(
        dist=CAMERA_DISTANCE,
        elev=elev,
        azim=azim,
        device=device,
    )

    return FoVPerspectiveCameras(
        device=device,
        R=R,
        T=T,
    )


def create_rasterizer_and_shader(device: torch.device):
    """
    创建老师参考实现使用的软剪影渲染管线。
    """
    cameras = create_cameras(device)

    blur_radius = float(np.log(1.0 / SIGMA - 1.0) * SIGMA)

    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=IMAGE_SIZE,
            blur_radius=blur_radius,
            faces_per_pixel=FACES_PER_PIXEL,
        ),
    )

    shader = SoftSilhouetteShader(
        blend_params=BlendParams(
            sigma=SIGMA,
            gamma=GAMMA,
        )
    )

    return rasterizer, shader


@torch.no_grad()
def render_target_silhouette(cow_mesh, rasterizer, shader) -> torch.Tensor:
    """
    渲染目标奶牛的多视角 alpha 剪影。
    """
    cow_mesh_batch = cow_mesh.extend(NUM_VIEWS)
    return shader(
        rasterizer(cow_mesh_batch),
        cow_mesh_batch,
    )[..., 3].detach()


def render_predicted_silhouette(src_mesh, rasterizer, shader) -> torch.Tensor:
    """
    渲染当前源网格的多视角 alpha 剪影。
    """
    src_mesh_batch = src_mesh.extend(NUM_VIEWS)
    return shader(
        rasterizer(src_mesh_batch),
        src_mesh_batch,
    )[..., 3]
