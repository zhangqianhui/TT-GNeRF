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
import legacy
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from torch.nn import functional as F
import functools
import copy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch.utils import data
from torch import autograd
from torchvision import transforms, utils
from dataset import FFHQFake
from torch_utils import misc
from kmeans.factor_catalog import FactorCatalog
#----------------------------------------------------------------------------
import cv2

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(2022)

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

def make_label(batch, latent_dim, device):
    labels = torch.randint(0, 2, (batch, latent_dim), device=device).float()
    return labels

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        if 'style_editing' in k:
            par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def calculate_average_coordinate(binary_mask):

    non_zero_coords = np.transpose(np.nonzero(binary_mask))
    if len(non_zero_coords) > 0:
        avg_coord = np.mean(non_zero_coords, axis=0)
        return tuple(avg_coord)
    else:
        return None

def training_steps(
    network_pkl,
    outdir,
    num_steps,
    fov_deg,
    model_size,
    dataset_path,
    csvpath,
    common_kwargs,
    G_kwargs,
    file_id
):

    device = torch.device('cuda')
    G_params = torch.load(network_pkl)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((model_size, model_size))])

    dataset = FFHQFake(dataset_path, transform, model_size, csvpath)

    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    if network_pkl is not None:
        print(f'Resuming from "{network_pkl}"')
        for name, module in [('G', G), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(G_params[name], module, require_all=True)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    outdir = os.path.join(outdir, str(file_id))

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    w = torch.from_numpy(dataset[file_id][2]).unsqueeze(0)
    w_o = w.to(device)

    k = 3
    catalog3 = FactorCatalog(k, random_state=10, batch_size=100,
                          n_init=3)

    # expression and beard
    k = 4
    catalog4 = FactorCatalog(k, random_state=10, batch_size=100,
                          n_init=3)

    angle_p = 0.0
    angle_y = 0.0

    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    output = G.synthesis(w_o, camera_params, neural_rendering_resolution=128)

    feature_image = output['feature_image']
    original_img = output['image']

    output3 = catalog3.fit_predict(feature_image.cpu(), raw=True)
    heatmaps3 = output3.get(feature_image.shape[-1])

    output4 = catalog4.fit_predict(feature_image.cpu(), raw=True)
    heatmaps4 = output4.get(feature_image.shape[-1])

    combined_heatmap3 = np.zeros((512, 512, 1), dtype=np.uint8)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
              (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
              (128, 0, 128), (0, 128, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64),
              (64, 64, 0), (64, 0, 64), (0, 64, 64), (192, 0, 0), (0, 192, 0)]

    scale_factor_x = combined_heatmap3.shape[0] / heatmaps3.shape[2]
    scale_factor_y = combined_heatmap3.shape[1] / heatmaps3.shape[3]

    name = ['face', 'hair', 'background']
    flag1 = 0
    flag2 = 0
    for i in range(heatmaps3.shape[1]):

        heatmap3 = heatmaps3[0, i].numpy()
        scaled_heatmap = heatmap3.repeat(scale_factor_x, axis=0).repeat(scale_factor_y, axis=1)
        combined_heatmap3 = (np.expand_dims(scaled_heatmap, axis=2)).astype(np.uint8)
        combined_heatmap3_ = np.clip(combined_heatmap3 * 255.0, 0, 255).astype(np.uint8)
        if i == 0:
            combined_heatmap3_crop = combined_heatmap3[256-3:256+3,256-3:256+3,:]
            if np.sum(combined_heatmap3_crop) < 5:
                flag1 = 1
                PIL.Image.fromarray(combined_heatmap3_.repeat(3, axis=-1), 'RGB').save(
                    f'{outdir}/heatmap3_{name[1]}.png')
            else:
                flag1 = 0
                PIL.Image.fromarray(combined_heatmap3_.repeat(3, axis=-1), 'RGB').save(
                    f'{outdir}/heatmap3_{name[0]}.png')

        if i == 1:
            combined_heatmap3_crop = combined_heatmap3[0:5,0:5,:]
            print(i, np.sum(combined_heatmap3_crop))
            if np.sum(combined_heatmap3_crop) < 5:
                flag2 = 0
                if flag1 == 0:
                    PIL.Image.fromarray(combined_heatmap3_.repeat(3, axis=-1), 'RGB').save(
                        f'{outdir}/heatmap3_{name[1]}.png')
                else:
                    PIL.Image.fromarray(combined_heatmap3_.repeat(3, axis=-1), 'RGB').save(
                        f'{outdir}/heatmap3_{name[0]}.png')
            else:
                flag2 = 1

                PIL.Image.fromarray(combined_heatmap3_.repeat(3, axis=-1), 'RGB').save(
                    f'{outdir}/heatmap3_{name[2]}.png')

        if i == 2:
            if flag2 == 0 and flag1 == 0:
                PIL.Image.fromarray(combined_heatmap3_.repeat(3, axis=-1), 'RGB').save(
                    f'{outdir}/heatmap3_{name[2]}.png')
            elif flag2 == 1 and flag1 == 0:
                PIL.Image.fromarray(combined_heatmap3_.repeat(3, axis=-1), 'RGB').save(
                    f'{outdir}/heatmap3_{name[1]}.png')
            elif flag2 == 1 and flag1 == 1:
                PIL.Image.fromarray(combined_heatmap3_.repeat(3, axis=-1), 'RGB').save(
                    f'{outdir}/heatmap3_{name[0]}.png')
            elif flag2 == 0 and flag1 == 1:
                PIL.Image.fromarray(combined_heatmap3_.repeat(3, axis=-1), 'RGB').save(
                    f'{outdir}/heatmap3_{name[2]}.png')

    for i in range(heatmaps4.shape[1]):

        heatmap4 = heatmaps4[0, i].numpy()
        scaled_heatmap = heatmap4.repeat(scale_factor_x, axis=0).repeat(scale_factor_y, axis=1)
        combined_heatmap4 = (np.expand_dims(scaled_heatmap, axis=2)).astype(np.uint8)
        combined_heatmap4 = np.clip(combined_heatmap4 * 255.0, 0, 255).astype(np.uint8)
        PIL.Image.fromarray(combined_heatmap4.repeat(3, axis=-1), 'RGB').save(f'{outdir}/heatmap4_{i}.png')

    original_img = (original_img + 1) * (255 / 2)
    PIL.Image.fromarray(original_img.permute(0, 2, 3, 1)[0].clamp(0,255).cpu().to(torch.uint8).numpy(), 'RGB').save(f'{outdir}/_orginal.png')

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

    parser.add_argument('--file_id', help='file_id',
                        type=int, default = 0)

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

    training_steps(args.network_pkl, args.outdir, args.num_steps, args.fov_deg,
                   args.model_size, args.dataset_path, args.csvpath, common_kwargs,
                   G_kwargs, args.file_id)

#----------------------------------------------------------------------------
