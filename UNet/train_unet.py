import argparse
import os

import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model import UNet
from data_load import VideoDataset


def record_history(path, idx, loss):
    try:
        with open(path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow({idx, loss})
    except:
        f = open(path, 'w')
        f.write({idx, loss})
        f.close()

def train(args):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "2"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = f'{args.root}{args.ckpt}'
    try: os.mkdir(ckpt)
    except: pass
    dataset = VideoDataset(args)
    n_val = int(len(dataset) * args.split_ratio)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                generator=torch.Generator().manual_seed(0))
    loader_args = dict(batch_size=args.batch_size, num_workers=4,
                        pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    csv_path = f'{ckpt}/history.csv'

    model = UNet(n_channels=15, n_classes=3)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
    criterion = nn.MSELoss()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    torch.save(optimizer.state_dict(), f'{ckpt}/opt.pth')

    for epoch in range(args.epoch):
        running_loss = 0
        with tqdm(total=n_train,desc=f'Epoch {epoch+1}/{args.epoch}') as pbar:
            for i, batch in enumerate(train_loader):
                inputx, target = batch
                inputx, target = inputx.cuda(), target.cuda()
                inputx = inputx.to(device=device)
                target = target.to(device=device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=False):
                    output = model(inputx).float()
                    loss = criterion(output, target)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                running_loss += loss.item()
                print(f'Step: {i+1}, Running Loss: {loss.item()}')
                pbar.update(1)
        record_history(csv_path, epoch+1, running_loss/len(train_set))
        print(f'Epoch: {epoch+1}, Loss: {running_loss}')
        torch.save(model.state_dict(),ckpt+f'/{epoch+1}_{running_loss}.pth')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/eunu/vid_stab')
    parser.add_argument('--stable', type=str, default='/DeepStab/stable')
    parser.add_argument('--unstable', type=str, default='/DeepStab/unstable')
    parser.add_argument('--ckpt', type=str, default='/UNet/ckpt_finetune')
    parser.add_argument('--scale', type=float, default=0.1)
    parser.add_argument('--split_ratio', type=float, default=0.15)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train', type=str, default=True)
    parser.add_argument('--momentum', type=float, default=0.9)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    train(args)