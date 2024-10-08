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
from torch.nn import functional as F
import functools
import imageio
import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.dual_discriminator import DualDiscriminator, depth2normal
import lpips
from torch.utils import data
import copy
from torch import autograd
from module.models import AttributeClassifier, TransferModel
from dataset import FFHQFake
from torchvision import transforms, utils
import torch.nn as nn

ID_TO_CLS = [
    '5_o_Clock_Shadow', #0
    'Arched_Eyebrows', #1
    'Attractive',   # 2
    'Bags_Under_Eyes', # 3
    'Bald',  # 4
    'Bangs', # 5
    'Big_Lips', # 6
    'Big_Nose', # 7
    'Black_Hair', # 8
    'Blond_Hair', # 9
    'Blurry', # 10
    'Brown_Hair',  # 11
    'Bushy_Eyebrows', # 12
    'Chubby', # 13
    'Double_Chin', # 14
    'Eyeglasses', # 15
    'Goatee',   # 16
    'Gray_Hair', # 17
    'Heavy_Makeup', # 18
    'High_Cheekbones', # 19
    'Male', # 20
    'Mouth_Slightly_Open', # 21
    'Mustache',  # 22
    'Narrow_Eyes', # 23
    'No_Beard',    # 24
    'Oval_Face',   # 25
    'Pale_Skin',
    'Pointy_Nose',
    'Receding_Hairline',
    'Rosy_Cheeks',
    'Sideburns',
    'Smiling',
    'Straight_Hair',
    'Wavy_Hair',
    'Wearing_Earrings',
    'Wearing_Hat',
    'Wearing_Lipstick',
    'Wearing_Necklace',
    'Wearing_Necktie',
    'Young',
]
CLS_TO_ID = {v: k for k, v in enumerate(ID_TO_CLS)}

def parse_range(s: Union[str, List]) -> List[int]:
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


# ----------------------------------------------------------------------------

def make_transform(translate: Tuple[float, float], angle: float):
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


# ----------------------------------------------------------------------------

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
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True,
                                         max_size=None, xflip=False)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)  # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution  # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels  # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj)  # Be explicit about dataset size.
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
    # weights = torch.Tensor([0.46, 2.02, 0.09, 5.77, 0.10, 6.288, 6.54])

    # 7.34
    weights = torch.Tensor([0.249, 3.32, 0.05, 1.96, 1.54, 1.87])
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
        batch,
        truncation_psi,
        trunccutoff,
        outdir,
        num_steps,
        fov_deg,
        model_size,
        dataset_path,
        csvpath,
        common_kwargs,
        G_kwargs,
        loss_dict,
        w_dir,
        id,
        ref_id
):

    device = torch.device('cuda')
    camera_lookat_point = torch.tensor([0, 0, 0.2], device=device)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(
        device)  # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()
    G2 = copy.deepcopy(G)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((model_size, model_size))])

    dataset = FFHQFake(dataset_path, transform, model_size, csvpath)

    if network_pkl is not None:
        print(f'Resuming from "{network_pkl}"')
        with dnnlib.util.open_url(network_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('G_ema', G_ema), ('G',  G2)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    outdir = outdir + '_' + str(id) + '_' + str(ref_id)

    print(device)
    classifier = AttributeClassifier(device=device)
    classifier.to(device)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    original_label = torch.tensor([dataset[id][1]]).float()
    w = torch.from_numpy(dataset[id][2]).unsqueeze(0)
    ws = w.to(device)
    original_label = original_label.to(device)

    cam2world_pose = LookAtPoseSampler.sample(
        3.14 / 2, 3.14 / 2 - 0.05,
        camera_lookat_point, radius=2.7, device=device)

    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    img_ref1 = G2.synthesis(ws=ws, c=c, noise_mode='const', neural_rendering_resolution=128)['image']

    logits = classifier(img_ref1)
    criterion = nn.CrossEntropyLoss()

    indices = [8, 20, 4, -1, -9, 24]
    logits = logits[:,indices]
    logits[:,1:5] = 1 - logits[:,1:5]

    print(logits)

    print(original_label.dtype, logits.dtype)
    loss = criterion(logits, original_label)

    print(loss.item())

    #
    # PIL.Image.fromarray(editing_triot.permute(0, 2, 3, 1)[0].clamp(0, 255).cpu().to(torch.uint8).numpy(), 'RGB').save(
    #     f'{outdir}/{id}_orginal.png')

# ----------------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="EG3D inversion")
    parser.add_argument("--network_pkl", type=str,
                        default='/nfs/data_chaos/jzhang/dataset/pretrained/eg3d/ffhq512-128.pkl',
                        help="path to the network pkl")
    parser.add_argument("--outdir", type=str, default='./outputs/', help="path to the lmdb dataset")
    parser.add_argument("--w_dir", type=str, default='./outputs/', help="the save w path")
    parser.add_argument('--truncation_psi', type=float, default=1.0, help='truncation_psi')
    parser.add_argument("--iter", type=int, default=10000, help="total training iterations")
    parser.add_argument("--id", type=int, default=0, help="id")
    parser.add_argument("--ref_id", type=int, default=0, help="ref_id")
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
    parser.add_argument('--sr_noise_mode', help='Type of noise for superresolution', choices=['random', 'none'],
                        default='none')
    parser.add_argument('--density_reg', help='Density regularization strength.', type=float, default=0.25)
    parser.add_argument('--density_reg_p_dist', help='density regularization strength.', type=float, default=0.004)
    parser.add_argument('--reg_type', help='Type of regularization',
                        choices=['l1', 'l1-alt', 'monotonic-detach', 'monotonic-fixed', 'total-variation'],
                        default='l1')
    parser.add_argument('--decoder_lr_mul', help='decoder learning rate multiplier.', type=float, default=1)
    parser.add_argument('--sr_num_fp16_res', help='Number of fp16 layers in superresolution', type=int, default=4)
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
                        type=int, default=2)
    parser.add_argument('--label_dim', help='the labels vector of images', type=int, default=6)

    parser.add_argument('--label_lambda_d', help='weight for lambda d',
                        type=int, default=1)

    parser.add_argument('--label_lambda_g', help='weight for lambda g',
                        type=int, default=10)

    parser.add_argument('--label_cls_d', help='weight for cls d',
                        type=int, default=1)

    parser.add_argument('--num_steps', help='iterations',
                        type=int, default=10000)

    parser.add_argument('--label_w_recon', help='weight for recon w',
                        type=int, default=6)

    parser.add_argument('--label_image_recon', help='weight for recon image',
                        type=int, default=6)

    parser.add_argument('--r1', help='weight for lambda d r1',
                        type=int, default=10)

    parser.add_argument('--d_regularize', help='d_regularize',
                        type=bool, default=True)

    parser.add_argument('--trunccutoff', help='truncation_cutoff',
                        type=int, default=14)

    parser.add_argument('--is_weighted', help='whether using weightd cls',
                        type=bool, default=True)

    args = parser.parse_args()
    G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator',
                    block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())

    G_kwargs.channel_base = D_kwargs.channel_base = args.cbase
    G_kwargs.channel_max = D_kwargs.channel_max = args.cmax
    G_kwargs.mapping_kwargs.num_layers = args.map_depth
    G_kwargs.class_name = 'training.triplane.TriPlaneGenerator'
    G_kwargs.fused_modconv_default = 'inference_only'  # Speed up training by using regular convolutions instead of grouped convolutions.

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

    common_kwargs = dict(c_dim=25, label_dim=args.label_dim, img_resolution=args.resolution, img_channels=3)
    loss_dict = dict(label_lambda_d=args.label_lambda_d,
                     label_lambda_g=args.label_lambda_g,
                     r1=args.r1,
                     d_regularize=args.d_regularize,
                     label_w_recon=args.label_w_recon,
                     label_image_recon=args.label_image_recon,
                     label_cls_d=args.label_cls_d,
                     is_weighted=args.is_weighted)

    training_steps(args.network_pkl, args.batch, args.truncation_psi, args.trunccutoff, args.outdir, args.num_steps,
                   args.fov_deg,
                   args.model_size, args.dataset_path, args.csvpath, common_kwargs, G_kwargs,
                   loss_dict, args.w_dir, args.id, args.ref_id)  # pylint: disable=no-value-for-parameter
# ----------------------------------------------------------------------------
