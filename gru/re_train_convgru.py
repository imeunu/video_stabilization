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
from torchvision import transforms
from tqdm import tqdm

from convgru import ConvGRU
from dataloader import VideoDataset


def get_input(args, device, idx):
    imgs, input_imgs = [], []
    if args.ssuuu:
        imgs.append(Image.open(f'{args.hdd}{args.val}/stable/8/{idx}.png'))
        imgs.append(Image.open(f'{args.hdd}{args.val}/stable/8/{idx+1}.png'))
        imgs.append(Image.open(f'{args.hdd}{args.val}/unstable/8/{idx+2}.png'))
        imgs.append(Image.open(f'{args.hdd}{args.val}/unstable/8/{idx+3}.png'))
        imgs.append(Image.open(f'{args.hdd}{args.val}/unstable/8/{idx+4}.png'))
    else:
        imgs.append(Image.open(f'{args.hdd}{args.val}/unstable/8/{idx}.png'))
        imgs.append(Image.open(f'{args.hdd}{args.val}/unstable/8/{idx+1}.png'))
        imgs.append(Image.open(f'{args.hdd}{args.val}/unstable/8/{idx+2}.png'))
        imgs.append(Image.open(f'{args.hdd}{args.val}/unstable/8/{idx+3}.png'))
        imgs.append(Image.open(f'{args.hdd}{args.val}/unstable/8/{idx+4}.png'))
    for img in imgs:
        img = torch.from_numpy(VideoDataset.preprocess(img, args.scale))
        img = img.unsqueeze(0)
        img = img.to(device)
        input_imgs.append(img)
    input_imgs = torch.stack(input_imgs,dim=1).float()
    return input_imgs

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
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--ckpt', type=str, default='/home/eunu/VideoStab_Unet/vid_stab/ckpt/ckpt_w5_s20')
    parser.add_argument('--eval_pth', type=str, default='/home/eunu/vid_stab/prediction')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--scale', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--hdd', type=str, default='/hdd/DeepStab')
    parser.add_argument('--val', type=str, default='/validation')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--pth', type=int, default=13)
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--ssuuu', type=bool, default=True)
    parser.add_argument('--stable', type=str, default='/DeepStab/stable')
    parser.add_argument('--unstable', type=str, default='/DeepStab/unstable')
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--window', type=int, default=5)
    return parser.parse_args()

def re_train(args, device):
    try: os.mkdir(args.ckpt)
    except: pass
    dataset = VideoDataset(args)
    loader_args = dict(batch_size=args.batch_size, num_workers=4)
    train_loader = DataLoader(dataset, shuffle=False, **loader_args)
    csv_path = f'{args.ckpt}/history.csv'

    model = load_model(args, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    criterion = nn.MSELoss()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    torch.save(optimizer.state_dict(), args.ckpt+'/opt.pth')

    for epoch in range(args.pth + 1, args.epoch):
        running_loss = 0
        with tqdm(total=len(dataset),desc=f'Epoch {epoch+1}/{args.epoch}') as pbar:
            for i, batch in enumerate(train_loader):
                inputx, target = batch
                inputx, target = inputx.cuda(), target.cuda()
                inputx = inputx.to(device=device).float()
                target = target.to(device=device).float()
                optimizer.zero_grad()
                x, loss = [], 0
                for j in range(args.window):
                    x.append(inputx[:,j,:,:,:])
                x = torch.stack(x, dim=1)
                for t in range(args.seq_len - args.window + 1):
                    with torch.cuda.amp.autocast(enabled=False):
                        o = model(x)
                        if t != args.seq_len - args.window:
                            x0 = x[:,1,:,:,:]
                            x2 = x[:,3:5,:,:,:].squeeze()
                            x = torch.cat([x0,o,x2,inputx[:,t+args.window,:,:,:]],dim=0)
                            x = x.unsqueeze(0)
                        loss += criterion(o, target[:,t,:,:,:])
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                running_loss += loss.item()
                print(f'Step: {i+1}, Running Loss: {loss.item()}')
                # with torch.cuda.amp.autocast(enabled=False):
                #     output = model(inputx.float())
                #     loss = criterion(output, target)
                # grad_scaler.scale(loss).backward()
                # grad_scaler.step(optimizer)
                # grad_scaler.update()
                # running_loss += loss.item()
                # print(f'Step: {i+1}, Running Loss: {loss.item()}')
                pbar.update(1)
        record_history(csv_path, epoch+1, running_loss/len(dataset))
        print(f'Epoch: {epoch+1}, Loss: {running_loss}')
        torch.save(model.state_dict(),args.ckpt+f'/{epoch+1}_{running_loss}.pth')

def evaluate(args, device):
    try: os.mkdir(args.eval_pth)
    except: pass
    model = load_model(args, device)
    model.eval()
    for i in range(1300):
        input_img = get_input(args, device, i)
        with torch.no_grad():
            output = model(input_img)
        output = postprocess(output, args)
        if args.save:
            savepath = f'{args.eval_pth}/output{i}.jpg'
            cv2.imwrite(savepath, output)
            print(f'saved {i}.jpg')

def load_model(args, device):
    model = ConvGRU()
    filename = [filename for filename in os.listdir(args.ckpt) if filename.startswith(f'{args.pth}_')]
    ckpt_path = f'{args.ckpt}/{filename[0]}'
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    return model

def postprocess(output, args):
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor()
    ])
    output = tf(output.squeeze())
    output = torch.clamp(output,0,1)
    output = (output * 255).permute(1,2,0).numpy().astype(np.uint8)
    output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
    return output


if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.test: evaluate(args, device)
    else: re_train(args, device)