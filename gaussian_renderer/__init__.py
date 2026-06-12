import math

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from diff_gaussian_rasterization.dog import (
    GaussianRasterizationSettings as DoGSettings,
    GaussianRasterizer as DoGRasterizer,
)

from . import network_gui
from utils.sh_utils import eval_sh


def _rasterizer(settings_type, rasterizer_type, camera, pc, pipe,
                background, scaling_modifier):
    settings = settings_type(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx * 0.5),
        tanfovy=math.tan(camera.FoVy * 0.5),
        bg=background,
        scale_modifier=scaling_modifier,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing,
    )
    return rasterizer_type(raster_settings=settings)


def _inputs(camera, pc, pipe, scaling_modifier):
    points = torch.zeros_like(pc.get_xyz, requires_grad=True, device="cuda")
    points.retain_grad()

    if pipe.compute_cov3D_python:
        scales, rotations = None, None
        covariance = pc.get_covariance(scaling_modifier)
    else:
        scales, rotations, covariance = pc.get_scaling, pc.get_rotation, None

    if pipe.convert_SHs_python:
        shs = pc.get_features.transpose(1, 2).view(
            -1, 3, (pc.max_sh_degree + 1) ** 2)
        directions = pc.get_xyz - camera.camera_center.expand(
            pc.get_features.shape[0], -1)
        directions = directions / directions.norm(dim=1, keepdim=True)
        colors = torch.clamp_min(
            eval_sh(pc.active_sh_degree, shs, directions) + 0.5, 0.0)
        dc, shs = None, None
    else:
        dc, shs, colors = pc.get_features_dc, pc.get_features_rest, None

    return points, scales, rotations, covariance, dc, shs, colors


def _result(image, radii, depth, points, pc, camera, use_trained_exp):
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(camera.image_name)
        image = (
            torch.matmul(
                image.permute(1, 2, 0), exposure[:3, :3]
            ).permute(2, 0, 1)
            + exposure[:3, 3, None, None]
        )

    return {
        "render": image.clamp(0, 1),
        "viewspace_points": points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth,
    }


def render(camera, pc, pipe, background, scaling_modifier=1.0,
           scores=None, use_trained_exp=False):
    values = _inputs(camera, pc, pipe, scaling_modifier)
    points, scales, rotations, covariance, dc, shs, colors = values
    rasterizer = _rasterizer(
        GaussianRasterizationSettings, GaussianRasterizer, camera, pc, pipe,
        background, scaling_modifier)
    image, radii, depth = rasterizer(
        means3D=pc.get_xyz,
        means2D=points,
        dc=dc,
        shs=shs,
        colors_precomp=colors,
        opacities=pc.get_opacity,
        scores=torch.zeros_like(pc.get_opacity) if scores is None else scores,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=covariance,
    )
    return _result(
        image, radii, depth, points, pc, camera, use_trained_exp)


def render_DoG(camera, pc, pipe, background, scaling_modifier=1.0,
               use_trained_exp=False, dog_mask=None, use_dog=True):
    if use_dog and pipe.compute_cov3D_python:
        raise ValueError(
            "DoG rendering requires CUDA scale/rotation covariance computation.")

    values = _inputs(camera, pc, pipe, scaling_modifier)
    points, scales, rotations, covariance, dc, shs, colors = values
    rasterizer = _rasterizer(
        DoGSettings, DoGRasterizer, camera, pc, pipe,
        background, scaling_modifier)
    dog_opacity = pc.get_dog_opacity if use_dog else None
    dog_scaling = pc.get_dog_scaling if use_dog else None

    if dog_mask is not None:
        mask = dog_mask.to(
            device=dog_opacity.device,
            dtype=dog_opacity.dtype,
        ).reshape(-1, 1)
        dog_opacity = dog_opacity * mask

    image, radii, depth = rasterizer(
        means3D=pc.get_xyz,
        means2D=points,
        dc=dc,
        shs=shs,
        colors_precomp=colors,
        opacities=pc.get_opacity,
        scores=torch.zeros_like(pc.get_opacity),
        scales=scales,
        dog_opacities=dog_opacity,
        dog_scales=dog_scaling,
        rotations=rotations,
        cov3D_precomp=covariance,
    )
    return _result(
        image, radii, depth, points, pc, camera, use_trained_exp)


def render_training(iteration, switch_iteration, camera, pc, pipe, background,
                    use_trained_exp=False):
    renderer = render_DoG if iteration >= switch_iteration else render
    return renderer(
        camera, pc, pipe, background,
        use_trained_exp=use_trained_exp,
    )


__all__ = ["network_gui", "render", "render_DoG", "render_training"]
