# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union
from module.flow import cnf
import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm
from dataset import FFHQFake
from camera_utils import LookAtPoseSampler



# ----------------------------------------------------------------------------

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length / 2
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

# ----------------------------------------------------------------------------
def gen_interp_video(G, outut_dir: str, seed, w_frames=60 * 4, w_select=1
                     , psi=1.0, truncation_cutoff=14, cfg='FFHQ', image_mode='image',
                     gen_shapes=False, flow_modules=None, latent_dir=None, cnf_path=None, device=torch.device('cuda'), **video_kwargs):


    prior = cnf(512, flow_modules, 6, 1)
    prior = prior.eval()

    cnf_ckpt_path = os.path.join(cnf_path, 'modellarge10k_010000.pt')
    prior.load_state_dict(torch.load(cnf_ckpt_path))

    with torch.no_grad():

        camera_lookat_point = torch.tensor([0, 0, 0.2], device=device) if cfg == 'FFHQ' else torch.tensor([0, 0, 0], device=device)

        for seed in range(0, 10):

            zs = torch.from_numpy(np.random.RandomState(seed).randn(G.z_dim)).to(device).unsqueeze(0)
            cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, camera_lookat_point, radius=2.7,
                                                      device=device)
            intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
            c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            w = G.mapping(z=zs, c=c, truncation_psi=psi, truncation_cutoff=truncation_cutoff)

            labels_dim = 6
            random_label = torch.randint(0, 1, (1, labels_dim), device=device).float()
            original_label = random_label.clone()
            intensity = 1.0

            for i in range(labels_dim * 2 + 1):

                random_label_ = random_label.clone()

                label_str = ''
                if i < labels_dim:
                    random_label_[:, i] = intensity - random_label_[:, i]

                    label_str = str(random_label_[0, 0].cpu().numpy())
                    for item in range(random_label_.shape[1] - 1):
                        label_str += '_' + str(random_label_[0, item + 1].cpu().numpy())

                elif i < labels_dim * 2:
                    random_label_[:, i - labels_dim] = -intensity - random_label_[:, i - labels_dim]

                    label_str = str(random_label_[0, 0].cpu().numpy())
                    for item in range(random_label_.shape[1] - 1):
                        label_str += '_' + str(random_label_[0, item + 1].cpu().numpy())

                if i == labels_dim * 2:
                    label_str = '0_0_0'
                    random_label_[:, :] = 0.0


                # w = np.load(os.path.join(latent_dir, '{}_styles.npz'.format(str(8))))['arr_0']
                # w = torch.from_numpy(w)
                # w = w.to(device)

                approx21, _ = prior(w, original_label, torch.zeros(1, w.shape[1], 1).to(w))
                rev = prior(approx21, random_label_, torch.zeros(1, w.shape[1], 1).to(w), True)

                video_name = os.path.join(outut_dir, '{}_{}.mp4'.format(str(seed), label_str))

                video_out = imageio.get_writer(video_name, mode='I', fps=60, codec='libx264', **video_kwargs)

                all_poses = []
                imgs = []
                for frame_idx in tqdm(range(w_frames)):

                    yaw_range = 0.35

                    cam2world_pose = LookAtPoseSampler.sample(
                        3.14 / 2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (w_frames)), 3.14 / 2 - 0.05,
                        camera_lookat_point, radius=2.7, device=device)

                    all_poses.append(cam2world_pose.squeeze().cpu().numpy())
                    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
                    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                    img = G.synthesis(ws=rev[0], c=c, noise_mode='const')[image_mode][0]
                    imgs.append(img)

                    img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    video_out.append_data(img.permute(1,2,0).cpu().numpy())

                video_out.close()

            np.savez(os.path.join(outut_dir, '{}_{}_styles.npz'.format(str(seed), str(original_label.tolist()))),
                     w.cpu().numpy())

# ----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    if m := re.match(r'^(\d+)[x,](\d+)$', s):
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')


# ----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds', required=True)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=120)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--cfg', help='Config', type=click.Choice(['FFHQ', 'Cats']), required=False, metavar='STR',
              default='FFHQ', show_default=True)
@click.option('--image_mode', help='Image mode', type=click.Choice(['image', 'image_depth', 'image_raw']),
              required=False, metavar='STR', default='image', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float,
              help='Multiplier for depth sampling in volume rendering', default=1, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
@click.option('--w_select', type=int, help='Neural rendering resolution override', default=1, show_default=True)
@click.option('--shapes', type=bool, help='Gen shapes for shape interpolation', default=False, show_default=True)
@click.option('--flow_modules', type=str, help='Gen shapes for shape interpolation', default='512-512-512-512-512', show_default=True)
@click.option('--latent_dir', type=str, help='Gen shapes for shape interpolation', default='', show_default=True)
@click.option('--cnf_path', type=str, help='Gen shapes for shape interpolation', default='', show_default=True)

def generate_images(
        network_pkl: str,
        seeds: int,
        truncation_psi: float,
        truncation_cutoff: int,
        w_frames: int,
        outdir: str,
        cfg: str,
        image_mode: str,
        sampling_multiplier: float,
        nrr: Optional[int],
        w_select: Optional[int],
        shapes: bool,
        flow_modules: str,
        latent_dir: str,
        cnf_path: str
):

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    """Render a latent vector interpolation video.

    Examples:

    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    Animation length and seed keyframes:

    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.

    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.

    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    # with dnnlib.util.open_url(network_pkl) as f:
    #     G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
    G = torch.load(network_pkl)['G_ema'].to(device)
    G.eval()

    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(
        G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    if nrr is not None: G.neural_rendering_resolution = nrr

    if truncation_cutoff == 0:
        truncation_psi = 1.0  # truncation cutoff of 0 means no truncation anyways
    if truncation_psi == 1.0:
        truncation_cutoff = 14  # no truncation so doesn't matter where we cutoff

    gen_interp_video(G=G, outut_dir=outdir,
                     w_frames=w_frames, seed=seeds, psi=truncation_psi, w_select=w_select,
                     truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode, gen_shapes=shapes,
                     flow_modules=flow_modules, latent_dir=latent_dir, cnf_path=cnf_path)
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
