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
import argparse
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
from torch.nn import functional as F
import functools
import imageio

from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch.utils import data
from torch import autograd
from training.loss import RegionL1lossOp, RegionL1loss_Normal
from module.flow import cnf
from torchvision import transforms, utils
from dataset import FFHQFake

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

def gradient_torch(pred):

    D_dy1 = pred[:, 1:2, :] - pred[:, 0:1, :]
    D_dy = (pred[:, 2:, :] - pred[:, :-2, :]) / 2
    D_dy2 = pred[:, -1:, :] - pred[:, -2:-1, :]
    D_dy = torch.cat([D_dy1, D_dy, D_dy2], dim=1)

    D_dx1 = pred[:, :, 1:2] - pred[:, :, 0:1]
    D_dx = (pred[:, :, 2:] - pred[:, :, :-2]) / 2
    D_dx2 = pred[:, :, -1:] - pred[:, :, -2:-1]
    D_dx = torch.cat([D_dx1, D_dx, D_dx2], dim=2)

    return D_dx, D_dy

def depth2normal(depth):

    dx, dy = gradient_torch(depth * 255.0)
    normal = torch.stack([-dx, -dy, torch.ones_like(dx).to(depth.device)], -1)
    normal = normal / (((normal ** 2).sum(3, keepdim=True)) ** 0.5 + 1e-7)
    normal = (normal + 1) / 2.0

    return normal

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

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def init_dataset_kwargs(data):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def requires_grad_parts(model, flag=False):

    params_dict_g = dict(model.named_parameters())
    for key, value in params_dict_g.items():
        decoder_cond = ('style_editing' in key)
        if decoder_cond:
            value.requires_grad = flag

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(outputs=real_pred.sum(),
                               inputs=real_img,
                               create_graph=True)

    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def getClassificationLoss(is_weighted, weights=torch.Tensor([0.46, 2.02, 5.77, 0.65, 6.28, 6.54]), device=None):

    #weights = torch.Tensor([0.46, 2.02, 0.09, 5.77, 0.10, 6.288, 6.54])
    weights = torch.Tensor([1.97, 0.45, 0.09, 6.20, 0.11, 5.66, 5.51])
    if is_weighted:
        return functools.partial(F.binary_cross_entropy_with_logits, pos_weight=weights.to(device))
    else:
        return F.binary_cross_entropy_with_logits

def make_label(batch, latent_dim, device):
    labels = torch.randint(0, 2, (batch, latent_dim), device=device).float()
    return labels

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        if 'style_editing' in k:
            par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def training_steps(
    network_pkl,
    truncation_psi,
    latent_dir,
    outdir,
    num_steps,
    fov_deg,
    model_size,
    dataset_path,
    csvpath,
    common_kwargs,
    G_kwargs,
    D_kwargs,
    loss_dict,
    output_shape,
    finetune_id,
    scale,
    file_id,
    cnf_path,
    flow_modules,
    lambda_normal
):

    device = torch.device('cuda')
    G = torch.load(network_pkl)['G_ema'].to(device)
    G.eval()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((model_size, model_size))])

    dataset = FFHQFake(dataset_path, transform, model_size, csvpath)

    prior = cnf(512, flow_modules, 6, 1)
    prior = prior.eval()

    cnf_ckpt_path = os.path.join(cnf_path, 'modellarge10k_010000.pt')
    prior.load_state_dict(torch.load(cnf_ckpt_path))

    original_label = torch.tensor([dataset[file_id][1]]).float()
    w = torch.from_numpy(dataset[file_id][2]).unsqueeze(0)
    w_o = w.to(device)
    original_label = original_label.to(device)

    # loss
    regionl1 = RegionL1lossOp()
    regionl1n = RegionL1loss_Normal()
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    for stage in range(0, 3):

        print(original_label)

        if stage == 0:
            finetune_id = 4
            scale = -1.5
        elif stage == 1:
            finetune_id = 1
            scale = 1.5
            num_steps = 100

        elif stage == 2:
            finetune_id = 0
            scale = 1.5
            num_steps = 100


        outdir_  = outdir + '_' + str(file_id) + '_' + str(finetune_id) + '_' + str(scale) + '_' +  str(num_steps) + '_' + str(lambda_normal)

        if not os.path.exists(outdir_):
            os.mkdir(outdir_)

        # hair color 1: brown hair color
        # gender 1: female
        # bangs 1: remove bangs
        # age 1: old
        # Smile 1: non-smiling
        # Beard 1: beard

        finetune_id = finetune_id
        label_dim = 6

        random_label = torch.randint(0, 1, (1, label_dim), device=device).float()

        scale = scale

        if finetune_id < label_dim:

            random_label[:, finetune_id] = scale - random_label[:, finetune_id]
            label_str = str(random_label[0, 0].cpu().numpy())
            for item in range(random_label.shape[1] - 1):
                label_str += '_' + str(random_label[0, item + 1].cpu().numpy())

        elif finetune_id < label_dim * 2:
            random_label[:, finetune_id - label_dim] = -scale - random_label[:, finetune_id - label_dim]

            label_str = str(random_label[0, 0].cpu().numpy())
            for item in range(random_label.shape[1] - 1):
                label_str += '_' + str(random_label[0, item + 1].cpu().numpy())

        if finetune_id == label_dim * 2:

            label_str = '0_0_0'
            random_label[:, :] = 0.0

        approx21, _ = prior(w_o, original_label, torch.zeros(1, w_o.shape[1], 1).to(w_o))
        rev = prior(approx21, original_label + random_label, torch.zeros(1, w_o.shape[1], 1).to(w_o), True)
        w_o_ = rev[0]

        w_o_copy = w_o_.clone().detach()

        w_o_ = w_o_.detach()

        w_o_.requires_grad = True

        initial_learning_rate = 0.0005
        opti_w = torch.optim.Adam({w_o_}, betas=(0.9, 0.99), lr=initial_learning_rate)

        num_steps = num_steps

        for step in range(num_steps):

            angle_p = np.random.uniform(-0.5, 0.5, 1)[0]
            angle_y = np.random.uniform(-0.5, 0.5, 1)[0]

            cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
            cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
            cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            real_imgs_dict = G.synthesis(w_o, camera_params, neural_rendering_resolution=128)
            fake_imgs_dict = G.synthesis(w_o_, camera_params, neural_rendering_resolution=128)
            fake_imgs_init_dict = G.synthesis(w_o_copy, camera_params, neural_rendering_resolution=128)

            image_depth_o = real_imgs_dict['image_depth'].squeeze(1)
            image_depth_f = fake_imgs_dict['image_depth'].squeeze(1)
            image_depth_f_o = fake_imgs_init_dict['image_depth'].squeeze(1)

            image_normal_o = depth2normal(image_depth_o)
            image_normal_f = depth2normal(image_depth_f)
            image_normal_f_o = depth2normal(image_depth_f_o)

            if finetune_id in [0, 1, 3, 5]:
                loss_normal = torch.nn.L1Loss()(image_normal_o, image_normal_f)
            elif finetune_id == 4:
                loss_normal, _ = regionl1n(real_imgs_dict['image'], image_normal_o, image_normal_f_o, image_normal_f, finetune_id)
            elif finetune_id == 2:
                loss_normal, _ = regionl1n(real_imgs_dict['image'], image_normal_o, image_normal_f_o, image_normal_f, finetune_id)
            else:
                loss_normal = 0

            loss, image_mask = regionl1(real_imgs_dict['image'], real_imgs_dict['image'],
                                            fake_imgs_init_dict['image'], fake_imgs_dict['image'], finetune_id)

            loss = loss + loss_normal * lambda_normal

            opti_w.zero_grad()
            loss.backward()
            opti_w.step()

            if step % 10 == 0:

                print(f"step: {step} "
                    f"loss: {loss:.3f}"
                      f"loss_normal: {loss_normal:.3f}"
                )

            if step % 20 == 0:

                with torch.no_grad():

                    original_img = G.synthesis(w_o, camera_params, neural_rendering_resolution=128)['image']
                    editing_triot = G.synthesis(w_o_, camera_params, neural_rendering_resolution=128)['image']
                    editing_init = G.synthesis(w_o_copy, camera_params, neural_rendering_resolution=128)['image']

                    original_img = (original_img + 1) * (255 / 2)
                    editing_init = (editing_init + 1) * (255 / 2)
                    editing_triot = (editing_triot + 1) * (255 / 2)

                    PIL.Image.fromarray(original_img.permute(0, 2, 3, 1)[0].clamp(0,255).cpu().to(torch.uint8).numpy(), 'RGB').save(f'{outdir_}/{step:04d}_orginal.png')
                    PIL.Image.fromarray(editing_init.permute(0, 2, 3, 1)[0].clamp(0,255).cpu().to(torch.uint8).numpy(), 'RGB').save(f'{outdir_}/{step:04d}_editing_init.png')
                    PIL.Image.fromarray(editing_triot.permute(0, 2, 3, 1)[0].clamp(0,255).cpu().to(torch.uint8).numpy(), 'RGB').save(f'{outdir_}/{step:04d}_editing_triot.png')

                    if finetune_id != 1:

                        if image_mask.shape[1] == 1:

                            image_mask = image_mask.repeat(1, 3, 1, 1)
                        image_mask = image_mask * 255.0
                        PIL.Image.fromarray(image_mask.permute(0, 2, 3, 1)[0].clamp(0,255).cpu().to(torch.uint8).numpy(), 'RGB').save(f'{outdir_}/{finetune_id:04d}_image_mask.png')

        video_name_triot = os.path.join(outdir_, '{}_triot.mp4'.format(label_str))
        video_normal_triot = os.path.join(outdir_, '{}_normal_triot.mp4'.format(label_str))
        video_name_init = os.path.join(outdir_, '{}_init.mp4'.format(label_str))
        video_normal_init = os.path.join(outdir_, '{}_normal_init.mp4'.format(label_str))
        video_name_o = os.path.join(outdir_, '0_0_0.mp4')
        video_normal_o = os.path.join(outdir_, '{}_normal_o.mp4'.format(label_str))

        video_triot = imageio.get_writer(video_name_triot, mode='I', fps=60, codec='libx264')
        video_init = imageio.get_writer(video_name_init, mode='I', fps=60, codec='libx264')
        video_o = imageio.get_writer(video_name_o, mode='I', fps=60, codec='libx264')
        video_normal_o = imageio.get_writer(video_normal_o, mode='I', fps=60, codec='libx264')
        video_normal_init = imageio.get_writer(video_normal_init, mode='I', fps=60, codec='libx264')
        video_normal_triot = imageio.get_writer(video_normal_triot, mode='I', fps=60, codec='libx264')

        w_frames = 120
        max_batch = 100000
        voxel_resolution = 128
        psi = 0.7
        camera_lookat_point = torch.tensor([0, 0, 0.2], device=device)

        for frame_idx in tqdm(range(w_frames)):

            pitch_range = 0.25
            yaw_range = 0.35
            # cam2world_pose = LookAtPoseSampler.sample(
            #     3.14 / 2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (w_frames)),
            #     3.14 / 2 - 0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (w_frames)),
            #     camera_lookat_point, radius=2.7, device=device)

            cam2world_pose = LookAtPoseSampler.sample(
                3.14 / 2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (w_frames)), 3.14 / 2 - 0.05,
                camera_lookat_point, radius=2.7, device=device)

            intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
            c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            img = G.synthesis(ws=w_o_, c=c, noise_mode='const')['image'][0]
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            video_triot.append_data(img.permute(1, 2, 0).cpu().numpy())

            img = G.synthesis(ws=w_o_copy, c=c, noise_mode='const')['image'][0]
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            video_init.append_data(img.permute(1, 2, 0).cpu().numpy())

            img = G.synthesis(ws=w_o, c=c, noise_mode='const')['image'][0]
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            video_o.append_data(img.permute(1, 2, 0).cpu().numpy())

            img = G.synthesis(ws=w_o, c=c, noise_mode='const')['image_depth']
            normal_o = depth2normal(img.squeeze(1))
            normal_o = (normal_o.clamp(0, 1) * 255.0).to('cpu', torch.uint8).numpy()
            video_normal_o.append_data(normal_o[0])

            img = G.synthesis(ws=w_o_copy, c=c, noise_mode='const')['image_depth']
            normal_init = depth2normal(img.squeeze(1))
            normal_init = (normal_init.clamp(0, 1) * 255.0).to('cpu', torch.uint8).numpy()
            video_normal_init.append_data(normal_init[0])

            img = G.synthesis(ws=w_o_, c=c, noise_mode='const')['image_depth']
            normal_triot = depth2normal(img.squeeze(1))
            normal_triot = (normal_triot.clamp(0, 1) * 255.0).to('cpu', torch.uint8).numpy()
            video_normal_triot.append_data(normal_triot[0])

        if output_shape:

            if finetune_id == 0:
                w_name = ['w_o_', 'w_o']
                w_objects = [w_o_, w_o]
            else:
                w_name = ['w_o_']
                w_objects = [w_o_]
            for k, w in enumerate(w_objects):

                samples, voxel_origin, voxel_size = create_samples(N=voxel_resolution, voxel_origin=[0, 0, 0],
                                                                   cube_length=G.rendering_kwargs['box_warp'])
                samples = samples.to(device)
                sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
                transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
                transformed_ray_directions_expanded[..., -1] = -1

                head = 0
                with tqdm(total=samples.shape[1]) as pbar:
                    with torch.no_grad():
                        while head < samples.shape[1]:
                            torch.manual_seed(0)
                            sigma = G.sample_mixed(samples[:, head:head + max_batch],
                                                   transformed_ray_directions_expanded[:, :samples.shape[1] - head],
                                                   w, truncation_psi=psi, noise_mode='const')['sigma']
                            sigmas[:, head:head + max_batch] = sigma
                            head += max_batch
                            pbar.update(max_batch)

                sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
                sigmas = np.flip(sigmas, 0)

                pad = int(30 * voxel_resolution / 256)
                pad_top = int(38 * voxel_resolution / 256)
                sigmas[:pad] = 0
                sigmas[-pad:] = 0
                sigmas[:, :pad] = 0
                sigmas[:, -pad_top:] = 0
                sigmas[:, :, :pad] = 0
                sigmas[:, :, -pad:] = 0

                output_ply = True
                if output_ply:
                    from shape_utils import convert_sdf_samples_to_ply
                    convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1,
                                               os.path.join(outdir_, '{}_{}_shape.ply'.format(label_str, w_name[k])), level=10)
                else:  # output mrc
                    with mrcfile.new_mmap(outdir_ + f'{0:04d}_shape.mrc', overwrite=True, shape=sigmas.shape,
                                          mrc_mode=2) as mrc:
                        mrc.data[:] = sigmas

        video_triot.close()
        video_init.close()
        video_o.close()

        original_label = original_label + random_label
        w_o = w_o_


#----------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="EG3D inversion")
    parser.add_argument("--network_pkl", type=str, default='./networks/ffhq512-128.pkl', help="path to the network pkl")
    parser.add_argument("--outdir", type=str, default='./outputs/', help="path to the lmdb dataset")
    parser.add_argument('--truncation_psi', type=float, default=1.0, help='truncation_psi')
    parser.add_argument("--iter", type=int, default=10000, help="total training iterations")
    parser.add_argument("--fov_deg", type=float, default=18.837, help="total training iterations")
    parser.add_argument("--batch", type=int, default=16, help="batch sizes for each gpus")
    parser.add_argument("--model_size", type=int, default=512, help="model size")
    parser.add_argument("--dataset_path", type=str, default='./training_data_40000', help="path to the dataset path")
    parser.add_argument("--csvpath", type=str, default='./training_data_40000', help="path to csv file")
    parser.add_argument("--resolution", type=int, default=512, help="resolution")
    parser.add_argument('--gen_pose_cond', help='If true, enable generator pose conditioning.', type=bool, default=True)
    parser.add_argument('--gpc_reg_prob',
                  help='Strength of swapping regularization. None means no generator pose conditioning, i.e. condition with zeros.'
                        , type=float, default=0.5)
    parser.add_argument('--sr_noise_mode', help='Type of noise for superresolution', choices=['random', 'none'], default='none')
    parser.add_argument('--density_reg', help='Density regularization strength.', type=float, default=0.25)
    parser.add_argument('--density_reg_p_dist', help='density regularization strength.', type=float, default=0.004)
    parser.add_argument('--reg_type', help='Type of regularization', choices=['l1', 'l1-alt', 'monotonic-detach', 'monotonic-fixed', 'total-variation'], default='l1')
    parser.add_argument('--decoder_lr_mul', help='decoder learning rate multiplier.', type=float, default=1)
    parser.add_argument('--sr_num_fp16_res', help='Number of fp16 layers in superresolution',  type=int, default=4)
    parser.add_argument('--cbase', help='Capacity multiplier', type=int, default=32768)
    parser.add_argument('--cmax', help='Max. feature maps', type=int, default=512)
    parser.add_argument('--g_num_fp16_res', help='Number of fp16 layers in generator', type=int, default=0)
    parser.add_argument('--freezed', help='Freeze first layers of D', type=int, default=0)
    parser.add_argument('--disc_c_noise',
                  help='Strength of discriminator pose conditioning regularization, in standard deviations.'
                        , type=float, default=0)
    parser.add_argument('--d_num_fp16_res', help='Number of fp16 layers in discriminator',
                  type=int, default=4)
    parser.add_argument('--map_depth', help='Mapping network depth  [default: varies]',
                        type=int, default = 2)
    parser.add_argument('--label_dim', help='the labels vector of images', type=int, default=6)

    parser.add_argument('--label_lambda_d', help='weight for lambda d',
                        type=int, default = 1)

    parser.add_argument('--label_lambda_g', help='weight for lambda g',
                        type=int, default = 10)

    parser.add_argument('--num_steps', help='iterations',
                        type=int, default = 10000)

    parser.add_argument('--label_w_recon', help='weight for recon w',
                        type=int, default = 6)

    parser.add_argument('--label_image_recon', help='weight for recon image',
                        type=int, default= 6)

    parser.add_argument('--r1', help='weight for lambda d r1',
                        type=int, default = 10)

    parser.add_argument('--lambda_normal', help='weight for normal',
                        type=float, default = 10)

    parser.add_argument('--d_regularize', help='d_regularize',
                        type=bool, default = True)

    parser.add_argument('--trunccutoff', help='truncation_cutoff',
                        type=int, default = 14)

    parser.add_argument('--latent_dir', help='truncation_cutoff',
                        type=str, default = '')

    parser.add_argument('--output_shape', help='output the face shape',
                        type=bool, default = False)

    parser.add_argument('--finetune_id', help='finetune_id',
                        type=int, default = 14)

    parser.add_argument('--file_id', help='file_id',
                        type=int, default = 0)

    parser.add_argument('--scale', help='scale',
                        type=float, default =1.0)
    parser.add_argument('--cnf_path', help='cnf_path',
                        type=str, default = '')
    parser.add_argument('--flow_modules', type=str, help='Gen shapes for shape interpolation', default='512-512-512-512-512')

    args = parser.parse_args()

    G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator',
                    block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())

    G_kwargs.channel_base = D_kwargs.channel_base = args.cbase
    G_kwargs.channel_max = D_kwargs.channel_max = args.cmax
    G_kwargs.mapping_kwargs.num_layers = args.map_depth
    G_kwargs.class_name = 'training.triplane.TriPlaneGenerator'
    G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.

    if args.resolution == 512:
        sr_module = 'training.superresolution.SuperresolutionHybrid8XDC'
    elif args.resolution == 256:
        sr_module = 'training.superresolution.SuperresolutionHybrid4X'
    elif args.resolution == 128:
        sr_module = 'training.superresolution.SuperresolutionHybrid2X'
    else:
        assert False, f"Unsupported resolution {args.resolution}; make a new superresolution module"

    rendering_options = {
        'image_resolution': args.resolution,
        'disparity_space_sampling': False,
        'clamp_mode': 'softplus',
        'superresolution_module': sr_module,
        'c_gen_conditioning_zero': not args.gen_pose_cond,
        # if true, fill generator pose conditioning label with dummy zero vector
        'gpc_reg_prob': args.gpc_reg_prob if args.gen_pose_cond else None,
        'superresolution_noise_mode': args.sr_noise_mode,
        # [random or none], whether to inject pixel noise into super-resolution layers
        'density_reg': args.density_reg,  # strength of density regularization
        'density_reg_p_dist': args.density_reg_p_dist,
        # distance at which to sample perturbed points for density regularization
        'reg_type': args.reg_type,  # for experimenting with variations on density regularization
        'decoder_lr_mul': args.decoder_lr_mul,  # learning rate multiplier for decoder
        'sr_antialias': True,
    }

    rendering_options.update({
        'depth_resolution': 48,  # number of uniform samples to take per ray.
        'depth_resolution_importance': 48,  # number of importance samples to take per ray.
        'ray_start': 2.25,  # near point along each ray to start taking samples.
        'ray_end': 3.3,  # far point along each ray to stop taking samples.
        'box_warp': 1,
        # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
        'avg_camera_radius': 2.7,  # used only in the visualizer to specify camera orbit radius.
        'avg_camera_pivot': [0, 0, 0.2],  # used only in the visualizer to control center of camera rotation.
    })

    G_kwargs.rendering_kwargs = rendering_options
    G_kwargs.num_fp16_res = 0
    G_kwargs.sr_num_fp16_res = args.sr_num_fp16_res

    G_kwargs.sr_kwargs = dnnlib.EasyDict(channel_base=args.cbase, channel_max=args.cmax, fused_modconv_default='inference_only')
    G_kwargs.num_fp16_res = args.g_num_fp16_res
    G_kwargs.conv_clamp = 256 if args.g_num_fp16_res > 0 else None


    D_kwargs.block_kwargs.freeze_layers = args.freezed
    D_kwargs.epilogue_kwargs.mbstd_group_size = None
    D_kwargs.class_name = 'training.dual_discriminator.DualDiscriminator'
    D_kwargs.disc_c_noise = args.disc_c_noise # Regularization for discriminator pose conditioning

    D_kwargs.num_fp16_res = args.d_num_fp16_res
    D_kwargs.conv_clamp = 256 if args.d_num_fp16_res > 0 else None

    common_kwargs = dict(c_dim=25, label_dim=args.label_dim, img_resolution=args.resolution, img_channels=3)

    loss_dict = dict(label_lambda_d=args.label_lambda_d,
                     label_lambda_g=args.label_lambda_g,
                     r1 = args.r1,
                     d_regularize = args.d_regularize,
                     label_w_recon = args.label_w_recon,
                     label_image_recon = args.label_image_recon
                     )

    print('haha', args.dataset_path)

    training_steps(args.network_pkl, args.truncation_psi, args.latent_dir, args.outdir, args.num_steps, args.fov_deg,
                   args.model_size, args.dataset_path, args.csvpath, common_kwargs,
                   G_kwargs, D_kwargs, loss_dict, args.output_shape,
                   args.finetune_id, args.scale, args.file_id, cnf_path=args.cnf_path, flow_modules=args.flow_modules, lambda_normal=args.lambda_normal) # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
