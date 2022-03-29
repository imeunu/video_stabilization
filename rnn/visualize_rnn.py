import argparse
import os

import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn

from model import RNN


def postprocess(img):
    img = torch.clamp(img,0,1).squeeze().cpu()
    img = (img * 255).permute(1,2,0).numpy().astype(np.uint8)
    return img

def save_frame(args, img, idx):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,dsize=(args.width,args.height),interpolation=cv2.INTER_CUBIC)
    save_path = f'{args.plot_path}/{str(idx).zfill(4)}.jpg'
    cv2.imwrite(save_path, img)
    print(f'saved {str(idx).zfill(4)}.jpg')

def to_tensor(img):
    img = torch.tensor(img / 255)
    img = img.permute(0,3,1,2).contiguous()
    return img.float()

def load_model(args, device):
    model = RNN()
    model = nn.DataParallel(model)
    ckpt_path = f'{args.ckpt_read}/{str(args.pth).zfill(3)}.pth'
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    return model

def visualize(args):
    # Video number 54 to 61
    h5_path = f'/hdd/DeepStab/h5/{args.video_number}.h5'
    try: os.mkdir(args.plot_path)
    except: pass
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args, device).eval()
    
    with h5py.File(h5_path, 'r') as f:
        unstable = np.array(f['unstable'])
        unstable = to_tensor(unstable).to(device)

    with torch.no_grad():
        for i in range(len(unstable)):
            if not i:
                h = torch.zeros(1,20,18,32)
            output, h = model(unstable[i].unsqueeze(0), h)
            output = postprocess(output)
            save_frame(args, output, i)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--cuda', type=str, default='2')
    parser.add_argument('--plot_path', type=str, default='/home/eunu/vid_stab/savehere')
    parser.add_argument('--hidden_dim', type=int, default=80)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=180)

    parser.add_argument('--video_number', type=int, default=58)
    parser.add_argument('--pth', type=int, default=109)
    parser.add_argument('--ckpt_read', type=str, default='/home/eunu/vid_stab/ckpt/flow')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    visualize(args)