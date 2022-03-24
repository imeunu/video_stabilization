# import argparse
# import os
# import sys

# import torch
# import torch.nn.functional as F

# from opticalflow.core.raft import RAFT
# from opticalflow.core.update import BasicUpdateBlock
# from opticalflow.core.extractor import BasicEncoder
# from opticalflow.core.corr import CorrBlock
# from opticalflow.core.utils.utils import bilinear_sampler, coords_grid, upflow8


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', help="restore checkpoint", default='/home/eunu/vid_stab/opticalflow/models/raft-things.pth')
#     parser.add_argument('--path', help="dataset for evaluation", default='demo-frames')
#     parser.add_argument('--small', action='store_true', help='use small model')
#     parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
#     parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

#     parser.add_argument('--cuda', default='2')
#     # parser.add_argument('--corr_levels')
#     return parser.parse_args()

# def get_normalized_randn(*size):
#     rand = torch.randn(size)
#     rand = F.normalize(rand, p=2.0)
#     return rand.contiguous()

# def initialize_flow(img):
#     """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
#     N, C, H, W = img.shape
#     coords0 = coords_grid(N, H//8, W//8, device=img.device)
#     coords1 = coords_grid(N, H//8, W//8, device=img.device)

#     # optical flow computed as difference: flow = coords1 - coords0
#     return coords0, coords1

# if __name__ == '__main__':
#     args = get_args()
#     os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda
#     device = torch.device('cuda')
#     autocast = torch.cuda.amp.autocast

#     # corr_levels, corr_radius, mixed_precision, alter_corr = 4, 4, False, False
#     model = torch.nn.DataParallel(RAFT(args))
#     model.load_state_dict(torch.load(args.model))
#     # model = model.module
#     # model.to(device)
#     # model.eval()

#     flow_init, upsample, test_mode, iters = None, True, False, 20

#     i1, i2 = get_normalized_randn(4,3,176,320), get_normalized_randn(4,3,176,320)
#     # i1, i2 = i1.to(device), i2.to(device)
#     hdim, cdim = 128, 128
#     # i1, i2 size (4,3,180,320)
#     # _, flow = model(i1, i2)

#     fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=False)        
#     cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=False)
#     update_block = BasicUpdateBlock(args, hidden_dim=hdim)

#     with autocast(enabled=True):
#         fmap1, fmap2 = fnet([i1, i2])
#         # fmap1, fmap2 size: (4,256,23,40)
#         fmap1, fmap2 = fmap1.float(), fmap2.float()
#     corr_fn = CorrBlock(fmap1, fmap2, radius=args.corr_radius)
    
#     with autocast(enabled=args.mixed_precision):
#         cmap = cnet(i1)
#         net, inp = torch.split(cmap, [hdim, cdim], dim=1)
#         net = torch.tanh(net)
#         inp = torch.relu(inp)
#         # net, inp size: (4,128,23,40)
    
#     coords0, coords1 = initialize_flow(i1)
#     # coords size: (4,2,22,40)

#     flow_predictions = []
#     for itr in range(iters):
#         coords1 = coords1.detach()
#         coor = corr_fn(coords1)
#         # if up_mask is None:

import argparse
import os
import sys

import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torchvision.models.optical_flow import raft_small
# from tqdm import tqdm

from opticalflow.core.raft import RAFT
from opticalflow.core.utils.utils import InputPadder
from rnn.model import RNN
from rnn.rnn_loader import VideoDataset



def record_history(path, idx, loss):
    try:
        with open(path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow({idx, loss})
    except:
        f = open(path, 'w')
        f.write({idx, loss})
        f.close()

def load_model(args, device):
    model = RNN(hidden_dim=args.hidden_dim)
    model = nn.DataParallel(model)
    ckpt_path = f'{args.ckpt_read}/{str(args.pth).zfill(3)}.pth'
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    return model

def load_optical_flow(args, device):
    model = RAFT(args)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.flow_path))
    model = model.module
    model.to(device)
    return model.eval()

def init_all(model, init_func):
    for p in model.parameters():
        init_func(p)

def train(args):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try: os.mkdir(args.ckpt_save)
    except: pass
    # tf = transforms.Compose([])
    trainset = VideoDataset(args, train='all')
    trainset = DataLoader(trainset, shuffle=True, batch_size=args.batch_size,
                        num_workers=8, pin_memory=True)
    csv_path = f'{args.ckpt_save}/history.csv'
    # writer = SummaryWriter('runs/DeepStab')
    model = RNN(hidden_dim=args.hidden_dim)
    model.to(device)
    model = nn.DataParallel(model)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    torch.save(optimizer.state_dict(), f'{args.ckpt_save}/opt.pth')

    if args.re_train:
        model = load_model(args, device)
        for _ in range(args.pth):
            scheduler.step()
    else:
        args.pth = 0
    criterion = nn.MSELoss()
    
    if args.flow_loss:
        optflow = load_optical_flow(args, device)
        # flow_loss = 0
    flow_target_size = [args.batch_size, 2, int(8*np.ceil(args.img_size[0]/8)),
                        int(8*np.ceil(args.img_size[1]/8))]
    flow_target = torch.zeros(flow_target_size)
    flow_target = flow_target.to(device)
    padder = InputPadder(flow_target_size)

    for epoch in range(args.pth, args.epoch):
        running_loss = 0
        with tqdm(total=len(trainset), desc=f'Epoch {epoch+1}/{scheduler.get_last_lr()}') as pbar:
            for i, batch in enumerate(trainset):
                inputx, target = batch
                inputx, target, loss = inputx.cuda(), target.cuda(), 0
                inputx = inputx.to(device=device).float()
                target = target.to(device=device).float()
                optimizer.zero_grad()
                # inputx, target size (1,seq_len,3,72,128), (1,seq_len,3,72,128)
                for j in range(inputx.size(1)):
                    if not j:
                        h = torch.zeros(inputx.size(0),20,45,80)
                    output, h = model(inputx[:,j,:,:,:], h)
                    loss += criterion(output, target[:,j,:,:,:])
                    if args.flow_loss and not j: # flow_loss and j=0
                        prev = padder.pad(output)[0]
                    elif args.flow_loss and j:
                        output = padder.pad(output)[0]
                        flow = optflow(prev, output, iters=5, test_mode=True)
                        loss += args.lambda_flow * criterion(flow, flow_target)
                        prev = output
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                running_loss += loss.item()
                print(f'Step: {i+1}, Running Loss: {loss.item()}')
                pbar.update(1)
        record_history(csv_path, epoch+1, running_loss/len(trainset))
        # writer.add_scalar('training loss', running_loss/len(trainset))
        print(f'Epoch: {epoch+1}, Loss: {running_loss/len(trainset)}')
        scheduler.step()
        torch.save(model.state_dict(), args.ckpt_save + f'/{str(epoch+1).zfill(3)}.pth')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--ckpt_read', type=str, default='/home/eunu/vid_stab/ckpt/flow')
    parser.add_argument('--ckpt_save', type=str, default='/home/eunu/vid_stab/ckpt/flow')
    parser.add_argument('--flow_path', type=str, default='/home/eunu/vid_stab/opticalflow/models/raft-small.pth')
    parser.add_argument('--hidden_dim', type=int, default=80)
    parser.add_argument('--ssuuu', type=bool, default=False)
    parser.add_argument('--img_size', default=(180,320))
    parser.add_argument('--dataset', type=str, default='vr')

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--augmentation', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--lambda_flow', type=float, default=1)
    parser.add_argument('--flow_loss', type=bool, default=False)

    parser.add_argument('--pth', type=int, default=4)
    parser.add_argument('--re_train', type=bool, default=False)
    parser.add_argument('--cuda', type=str, default='1')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    train(args)