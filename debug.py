import argparse
import os
import sys

import torch
import torch.nn.functional as F

from opticalflow.core.raft import RAFT
from opticalflow.core.update import BasicUpdateBlock
from opticalflow.core.extractor import BasicEncoder
from opticalflow.core.corr import CorrBlock
from opticalflow.core.utils.utils import bilinear_sampler, coords_grid, upflow8


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default='/home/eunu/vid_stab/opticalflow/models/raft-things.pth')
    parser.add_argument('--path', help="dataset for evaluation", default='demo-frames')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    parser.add_argument('--cuda', default='2')
    # parser.add_argument('--corr_levels')
    return parser.parse_args()

def get_normalized_randn(*size):
    rand = torch.randn(size)
    rand = F.normalize(rand, p=2.0)
    return rand.contiguous()

def initialize_flow(img):
    """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
    N, C, H, W = img.shape
    coords0 = coords_grid(N, H//8, W//8, device=img.device)
    coords1 = coords_grid(N, H//8, W//8, device=img.device)

    # optical flow computed as difference: flow = coords1 - coords0
    return coords0, coords1

if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda
    device = torch.device('cuda')
    autocast = torch.cuda.amp.autocast

    # corr_levels, corr_radius, mixed_precision, alter_corr = 4, 4, False, False
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    # model = model.module
    # model.to(device)
    # model.eval()

    flow_init, upsample, test_mode, iters = None, True, False, 20

    i1, i2 = get_normalized_randn(4,3,176,320), get_normalized_randn(4,3,176,320)
    # i1, i2 = i1.to(device), i2.to(device)
    hdim, cdim = 128, 128
    # i1, i2 size (4,3,180,320)
    # _, flow = model(i1, i2)

    fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=False)        
    cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=False)
    update_block = BasicUpdateBlock(args, hidden_dim=hdim)

    with autocast(enabled=True):
        fmap1, fmap2 = fnet([i1, i2])
        # fmap1, fmap2 size: (4,256,23,40)
        fmap1, fmap2 = fmap1.float(), fmap2.float()
    corr_fn = CorrBlock(fmap1, fmap2, radius=args.corr_radius)
    
    with autocast(enabled=args.mixed_precision):
        cmap = cnet(i1)
        net, inp = torch.split(cmap, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        # net, inp size: (4,128,23,40)
    
    coords0, coords1 = initialize_flow(i1)
    # coords size: (4,2,22,40)

    flow_predictions = []
    for itr in range(iters):
        coords1 = coords1.detach()
        coor = corr_fn(coords1)
        # if up_mask is None: