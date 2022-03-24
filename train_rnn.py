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
from tqdm import tqdm

from opticalflow.core.utils.utils import InputPadder
from rnn.model import RNN
from rnn.rnn_loader import VideoDataset


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

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

def load_optical_flow(device):
    model = raft_small(pretrained=True)
    model = nn.DataParallel(model)
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
        optflow = load_optical_flow(device)
        # flow_loss = 0
    flow_target_size = (args.batch_size,2,int(8*np.ceil(args.img_size[0]/8)),
                        int(8*np.ceil(args.img_size[1]/8)))
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
                        prev = padder.unpad(output)[0]
                    elif args.flow_loss and j:
                        output = padder.unpad(output)[0]
                        flow = optflow(prev, output)
                        loss += args.lambda_flow * criterion(flow[-1], flow_target)
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
    parser.add_argument('--hidden_dim', type=int, default=80)
    parser.add_argument('--ssuuu', type=bool, default=False)
    parser.add_argument('--img_size', default=(180,320))
    parser.add_argument('--dataset', type=str, default='vr')

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--augmentation', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--lambda_flow', type=float, default=0.05)
    parser.add_argument('--flow_loss', type=bool, default=True)

    parser.add_argument('--pth', type=int, default=53)
    parser.add_argument('--re_train', type=bool, default=True)
    parser.add_argument('--cuda', type=str, default='1')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    temp = '1'
    train(args)