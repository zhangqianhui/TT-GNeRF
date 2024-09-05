# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import mrcfile
import cv2
from torch.nn import functional as F

import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator
import lpips
import json


#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=1024, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--file_id',  type=int, required=False, metavar='BOOL', default=False, show_default=True)
def inversion(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    class_idx: Optional[int],
    reload_modules: bool,
    file_id:int
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    # with dnnlib.util.open_url(network_pkl) as f:
    G = torch.load(network_pkl)['G_ema'].to(device) # type: ignore

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:

        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=False)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new

    os.makedirs(outdir, exist_ok=True)

    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device).eval()

    # real image

    # # Load VGG16 feature detector.
    # url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    # with dnnlib.util.open_url(url) as f:
    #     vgg16 = torch.jit.load(f).eval().to(device)

    # output_dir = './output_inversionv2/'

    file_id = file_id
    id_name = '{:0>5}.png'.format(file_id)

    target = cv2.imread(os.path.join("/nfs/data_chaos/jzhang/code/StyleSDF/FFHQ_inversion/aligned/", id_name))

    target = target[:,:,::-1].copy()

    with open("/nfs/data_chaos/jzhang/code/StyleSDF/FFHQ_inversion/aligned/dataset.json", 'rb') as f:
        json_data = json.load(f)['labels']

    pose_list = []
    for pose in json_data:
        if pose[0] == id_name:
            pose_list = pose[1]
            break

    camera_pose = torch.from_numpy(np.array(pose_list)).unsqueeze(0).to(device)

    target = torch.from_numpy(target).permute(2,0,1).unsqueeze(0).to(device).to(torch.float32) / 127.5 - 1.0
    target_ = F.interpolate(target, size=(512, 512), mode='area')

    noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}

    w_avg_samples = 1000
    # sampling z
    z = torch.from_numpy(np.random.RandomState(0).randn(w_avg_samples, G.z_dim)).to(device)

    angle_p = -0.2
    angle_y = 0.0
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
    camera_params = camera_pose.float()

    #conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    conditioning_params = camera_params
    conditioning_params = conditioning_params.repeat(z.shape[0],1)

    w_samples = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
    start_w = w_avg

    w_opt = torch.tensor(start_w, dtype=torch.float32, device=device,
                         requires_grad=True)

    first_inv_lr = 5e-3
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=first_inv_lr)

    num_steps = 1000
    lr_rampdown_length = 0.25
    lr_rampup_length = 0.05
    initial_learning_rate = 5e-3
    regularize_noise_weight = 1e5
    initial_noise_factor = 0.05
    noise_ramp_length = 0.75

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    pbar = range(num_steps)

    pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)

    for step in pbar:

        # Learning rate schedule.
        t = step / num_steps

        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2

        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)

        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        w_noise = torch.randn_like(w_opt) * w_noise_scale

        ws = (w_opt + w_noise).repeat([1, G.backbone.mapping.num_ws, 1])

        synth_images = G.synthesis(ws, camera_params, noise_mode='const')['image']
        #synth_images_ = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)

        # Features for synth images.
        lpips_loss = loss_fn_vgg(target_, synth_images).mean()
        reg_loss = reg_loss * regularize_noise_weight
        loss = lpips_loss + reg_loss

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

        if step % 10 == 0:
            pbar.set_description((f"step: {step} "
                f"g: {loss:.4f} "
                f"lpips: {lpips_loss:.4f} "
                f"reg_loss: {reg_loss:.4f} "
            ))

        if step % 100 == 0:

            with torch.no_grad():

                synth_images = G.synthesis(ws, camera_params, noise_mode='const')['image']
                synth_images = (synth_images + 1) * (255 / 2)
                synth_images = torch.clip(synth_images, 0, 255)

                PIL.Image.fromarray(synth_images.permute(0, 2, 3, 1)[0].cpu().to(torch.uint8).numpy(), 'RGB').save(f'{outdir}/{file_id}_{step:04d}_syn.png')
                PIL.Image.fromarray(((target + 1) * (255/2)).permute(0, 2, 3, 1)[0].cpu().to(torch.uint8).numpy(), 'RGB').save(f'{outdir}/{file_id}_{step:04d}_real.png')

    torch.save(w_opt, os.path.join(outdir, '{}_{}.pt'.format(file_id, num_steps)))
    torch.save(G, os.path.join(outdir, '{}_{}.pkl'.format(file_id, num_steps)))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    inversion() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
