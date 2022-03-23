import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

from data_load import VideoDataset
from model import UNet
from PIL import Image
from torchvision import transforms


def get_input(args, device, idx):
    imgs, input_imgs = [], []
    if args.ssuuu:
        imgs.append(Image.open(f'{args.root}{args.val}/stable/8/{idx}.png'))
        imgs.append(Image.open(f'{args.root}{args.val}/stable/8/{idx+1}.png'))
        imgs.append(Image.open(f'{args.root}{args.val}/unstable/8/{idx+2}.png'))
        imgs.append(Image.open(f'{args.root}{args.val}/unstable/8/{idx+3}.png'))
        imgs.append(Image.open(f'{args.root}{args.val}/unstable/8/{idx+4}.png'))
    else:
        imgs.append(Image.open(f'{args.root}{args.val}/unstable/8/{idx}.png'))
        imgs.append(Image.open(f'{args.root}{args.val}/unstable/8/{idx+1}.png'))
        imgs.append(Image.open(f'{args.root}{args.val}/unstable/8/{idx+2}.png'))
        imgs.append(Image.open(f'{args.root}{args.val}/unstable/8/{idx+3}.png'))
        imgs.append(Image.open(f'{args.root}{args.val}/unstable/8/{idx+4}.png'))
    for img in imgs:
        img = torch.from_numpy(VideoDataset.preprocess(img, args.scale))
        img = img.unsqueeze(0)
        img = img.to(device)
        input_imgs.append(img)
    input_imgs = torch.cat(input_imgs,dim=1).float()
    return input_imgs

def load_model(args, device):
    model = UNet(n_channels=15, n_classes=3)
    ckpt_path = os.path.join(args.root + args.ckpt, args.pth)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
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

def evaluate(args):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    for i in range(1300):
        input_img = get_input(args, device, i)
        with torch.no_grad():
            output = model(input_img)
        output = postprocess(output, args)
        if args.save:
            savepath = f'{args.root + args.savepath}{i}.jpg'
            cv2.imwrite(savepath, output)
            print(f'saved {i}.jpg')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/eunu/vid_stab')
    parser.add_argument('--val', type=str, default='/DeepStab/validation')
    parser.add_argument('--ckpt', type=str, default='/ckpt')
    parser.add_argument('--pth', type=str, default='22.pth')
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--savepath', type=str, default='/prediction/output')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--scale', type=float, default=0.1)
    parser.add_argument('--ssuuu', type=bool, default=True)
    parser.add_argument('--mkvid', type=bool, default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    evaluate(args)