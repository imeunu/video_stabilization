import argparse
import os
import sys

import csv
import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from UNet.unet_model import UNet
from dataloader import VideoDataset


def record_history(path, idx, loss):
    try:
        with open(path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow({idx, loss})
    except:
        f = open(path, 'w')
        f.write({idx, loss})
        f.close()

def save_frame(args, img, idx):
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor()
    ])
    img = torch.clamp(img,0,1)
    img = tf(img.squeeze())
    img = (img * 255).permute(1,2,0).numpy().astype(np.uint8)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # img = torch.clamp(img.squeeze(),0,1).cpu()
    # img = (img * 255).permute(1,2,0).detach().numpy().astype(np.uint8)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, dsize=(1280,720), interpolation=cv2.INTER_CUBIC)

    save_path = f'{args.plot_path}/{str(idx).zfill(4)}.jpg'
    cv2.imwrite(save_path, img)
    print(f'saved {str(idx).zfill(4)}.jpg')
    del img

def to_tensor(img):
    img = torch.tensor(img / 255)
    img = img.permute(0,3,1,2).contiguous()
    return img.float()

def load_model(args, device):
    model = UNet(n_channels=15, n_classes=3, hidden_dim=args.hidden_dim)
    model = nn.DataParallel(model)
    ckpt_path = f'{args.ckpt_read}/{str(args.pth).zfill(3)}.pth'
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    return model

def load_each_model(args, ckpt, device):
    model = UNet(n_channels=15, n_classes=3, hidden_dim=args.hidden_dim)
    model = nn.DataParallel(model)
    model.to(device)
    ckpt_path = f'{args.ckpt_read}/{ckpt}'
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    return model

def train(args):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try: os.mkdir(args.ckpt_save)
    except: pass
    trainset = VideoDataset(args, train=True)
    trainset = DataLoader(trainset, shuffle=True, batch_size=args.batch_size,
                        num_workers=4, pin_memory=True)
    csv_path = f'{args.ckpt_save}/history.csv'
    # writer = SummaryWriter('runs/DeepStab')
    if args.re_train:
        model = load_model(args, device)
    else:
        model = UNet(n_channels=15, n_classes=3, hidden_dim=args.hidden_dim)
        model.to(device)
        model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    criterion = nn.MSELoss()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    torch.save(optimizer.state_dict(), f'{args.ckpt_save}/opt.pth')

    for epoch in range(args.epoch):
        running_loss = 0
        with tqdm(total=len(trainset), desc=f'Epoch {epoch+1}/{args.epoch}') as pbar:
            for i, batch in enumerate(trainset):
                inputx, target = batch
                inputx, target = inputx.cuda(), target.cuda()
                inputx = inputx.to(device=device).float()
                target = target.to(device=device).float()
                optimizer.zero_grad()
                x, loss = [], 0
                # inputx, target size (1,60,72,128), (1,16,3,72,128)
                for j in range(args.window):
                    x.append(inputx[:,j,:,:,:])
                x = torch.cat(x,dim=1)
                for t in range(args.seq_len - args.window + 1):
                    with torch.cuda.amp.autocast(enabled=False):
                        o = model(x).float()
                        if t != args.seq_len - args.window:
                            x0 = x[:,3:6,:,:]
                            x2 = x[:,9:,:,:]
                            x = torch.cat([x0,o,x2,inputx[:,t+args.window,:,:,:]],dim=1)
                        loss += criterion(o,target[:,t,:,:,:])
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                running_loss += loss.item()
                print(f'Step: {i+1}, Running Loss: {loss.item()}')
                pbar.update(1)
        record_history(csv_path, epoch+1, running_loss/len(trainset))
        # writer.add_scalar('training loss', running_loss/len(trainset))
        print(f'Epoch: {epoch+1}, Loss: {running_loss/len(trainset)}')
        torch.save(model.state_dict(), args.ckpt_save + f'/{str(epoch+1).zfill(3)}.pth')

def measure(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = VideoDataset(args, train=False)
    testset = DataLoader(dataset, num_workers=2)
    criterion = torch.nn.MSELoss()
    csv_path, n = f'{args.ckpt_save}/measure.csv', 0
    while (n < args.epoch):
        for i, ckpt in enumerate(sorted(os.listdir(args.ckpt_read))):
            if i < n: continue
            try: model = load_each_model(args, ckpt, device)
            except: continue
            running_loss = 0; print(f'{n} Measuring {ckpt}...')
            with torch.no_grad():
                with tqdm(total=len(testset)) as pbar:
                    for batch in testset:
                        inputx, target = batch
                        inputx, target = inputx.cuda(), target.cuda()
                        inputx = inputx.to(device=device).float()
                        target = target.to(device=device).float()
                        x, loss = [], 0
                        for j in range(args.window):
                            x.append(inputx[:,j,:,:,:])
                        x = torch.cat(x,dim=1)
                        for t in range(args.seq_len - args.window + 1):
                            o = model(x)
                            if t != args.seq_len - args.window:
                                x0 = x[:,3:6,:,:]
                                x2 = x[:,9:,:,:]
                                x = torch.cat([x0,o,x2,inputx[:,t+args.window,:,:,:]],dim=1)
                            loss += criterion(o, target[:,t,:,:,:])
                        running_loss += loss.item()
                        pbar.update(1)
                record_history(csv_path, i+1, running_loss/len(testset))
            print(f'name: {ckpt}, test loss: {running_loss/len(testset)}')
            n += 1

def visualize(args):
    # Video number 54 to 61
    h5_path = f'/hdd/DeepStab/h5/{args.video_number}.h5'
    try: os.mkdir(args.plot_path)
    except: pass
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args, device)
    
    with h5py.File(h5_path, 'r') as f:
        if args.recurrent:
            ss = np.array(f['stable'])[:args.window//2]
            uuu = np.array(f['unstable'])[args.window//2:args.window]
            u = np.array(f['unstable'])[args.window:]
        else:
            unstable = np.array(f['unstable'])

    with torch.no_grad():
        if args.recurrent:
            ss, uuu, u, x = to_tensor(ss), to_tensor(uuu), to_tensor(u).to(device), []
            for frame in ss: x.append(frame.unsqueeze(0))
            for frame in uuu: x.append(frame.unsqueeze(0))
            x = torch.cat(x,dim=1).to(device)
            for i, frame in enumerate(u):
                o = model(x)
                save_frame(args, o, i)
                if i != len(u) - 1:
                    x0 = x[:,3:6,:,:]
                    x2 = x[:,9:,:,:]
                    x = torch.cat([x0,o,x2,frame.unsqueeze(0)],dim=1)
        else:
            unstable = to_tensor(unstable).to(device)
            for i in range(len(unstable)-args.window+1):
                x = unstable[i:i+args.window]
                x = torch.cat([frame for frame in x],dim=0).unsqueeze(0)
                o = model(x)
                save_frame(args, o, i)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--cuda', type=str, default='2')
    parser.add_argument('--ckpt_read', type=str, default='/home/eunu/vid_stab/ckpt/vr')
    parser.add_argument('--ckpt_save', type=str, default='/home/eunu/vid_stab/ckpt/vr')
    parser.add_argument('--plot_path', type=str, default='/home/eunu/vid_stab/savehere')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--scale', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--stable', type=str, default='/hdd/DeepStab/stable')
    parser.add_argument('--unstable', type=str, default='/hdd/DeepStab/unstable')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--ssuuu', type=bool, default=True)
    parser.add_argument('--pth', type=int, default=53)
    parser.add_argument('--re_train', type=bool, default=False)
    parser.add_argument('--video_number', type=int, default=58)
    parser.add_argument('--recurrent', type=bool, default=False)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    # train(args)
    # measure(args)
    visualize(args)