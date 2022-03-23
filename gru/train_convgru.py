import argparse
import os

import csv
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from gru.convgru import ConvGRU
from dataloader import VideoDataset

def record_history(path, idx, loss):
    try:
        with open(path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(idx, loss)
    except:
        # f = open(path, 'w')
        # f.write(f'{idx},{loss}\n')
        # f.close()
        pass

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--ckpt', type=str, default='/home/eunu/vid_stab/ckpt/')
    parser.add_argument('--split_ratio', type=float, default=0.15)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--scale', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--stable', type=str, default='/hdd/DeepStab/stable')
    parser.add_argument('--unstable', type=str, default='/hdd/DeepStab/unstable')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--ssuuu', type=bool, default=True)
    return parser.parse_args()


def train(args):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try: os.mkdir(args.ckpt)
    except: pass
    dataset = VideoDataset(args)
    loader_args = dict(batch_size=args.batch_size, num_workers=4)
    train_loader = DataLoader(dataset, shuffle=False, **loader_args)
    csv_path = f'{args.ckpt}/history.csv'

    model = ConvGRU().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    criterion = torch.nn.MSELoss()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    torch.save(optimizer.state_dict(), args.ckpt+'/opt.pth')

    for epoch in range(args.epoch):
        running_loss = 0
        with tqdm(total=len(dataset), desc=f'Epoch {epoch+1}/{args.epoch}') as pbar:
            for i, batch in enumerate(train_loader):
                inputx, target = batch
                inputx, target = inputx.cuda().float(), target.cuda().float()
                optimizer.zero_grad()
                x, loss = [], 0
                for j in range(args.window):
                    x.append(inputx[:,j,:,:,:])
                x = torch.stack(x, dim=1)
                for t in range(args.seq_len - args.window + 1):
                    with torch.cuda.amp.autocast(enabled=False):
                        o = model(x)
                        o.detach()
                        if t != args.seq_len - args.window:
                            x = x[:,args.window//2:,:,:,:].squeeze()
                            x = torch.cat([o,x,inputx[:,t+args.window,:,:,:]],dim=0)
                            x = x.unsqueeze(0)
                        loss += criterion(o, target[:,t,:,:,:])
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                running_loss += loss.item()
                print(f'Step: {i+1}, Running Loss: {loss.item()}')
    # for epoch in range(args.epoch):
    #     running_loss = 0
    #     with tqdm(total=len(dataset),desc=f'Epoch {epoch+1}/{args.epoch}') as pbar:
    #         for i, batch in enumerate(train_loader):
    #             inputx, target = batch
    #             inputx, target = inputx.cuda(), target.cuda()
    #             inputx = inputx.to(device=device)
    #             target = target.to(device=device)
    #             optimizer.zero_grad()
    #             with torch.cuda.amp.autocast(enabled=False):
    #                 output = model(inputx.float())
    #                 # print(output.size(), target.size())
    #                 loss = criterion(output, target)
    #             grad_scaler.scale(loss).backward()
    #             grad_scaler.step(optimizer)
    #             grad_scaler.update()
    #             running_loss += loss.item()
                pbar.update(1)
        record_history(csv_path, epoch+1, running_loss/len(dataset))
        print(f'Epoch: {epoch+1}, Loss: {running_loss}')
        torch.save(model.state_dict(),args.ckpt+f'/{epoch+1}_{running_loss}.pth')


if __name__ == '__main__':
    args = get_args()
    train(args)