import argparse
import os

import cv2
import h5py
import numpy as np
import torch
from torchvision import transforms

from gru.convgru import ConvGRU


def load_model(args, device):
    model = ConvGRU()
    filename = [filename for filename in os.listdir(args.ckpt) if filename.startswith(f'{args.pth}_')]
    ckpt_path = f'{args.ckpt}/{filename[0]}'
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    return model

def to_tensor(img):
    img = torch.tensor(img / 255)
    img = img.permute(0,3,1,2).contiguous()
    return img.float()

def save_frame(args, img, idx):
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((720, 1280)),
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

    save_path = f'{args.save_path}/{str(idx).zfill(4)}.jpg'
    cv2.imwrite(save_path, img)
    print(f'saved {str(idx).zfill(4)}.jpg')
    del img

def test(args):
    try: os.mkdir(args.save_path)
    except: pass
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args, device)
    
    with h5py.File(args.h5_path, 'r') as f:
        ss = np.array(f['stable'])[:args.window//2]
        uuu = np.array(f['unstable'])[args.window//2:args.window]
        u = np.array(f['unstable'])[args.window:]
    ss, uuu, u, x = to_tensor(ss), to_tensor(uuu), to_tensor(u).to(device), []

    for frame in ss: x.append(frame.unsqueeze(0))
    for frame in uuu: x.append(frame.unsqueeze(0))
    x = torch.stack(x,dim=1).to(device)
    for i, frame in enumerate(u):
        o = model(x)
        save_frame(args, o, i)
        if i != len(u) - 1:
            x0 = x[:,3:6,:,:]
            x2 = x[:,9:,:,:]
            x = torch.cat([x0,o,x2,frame.unsqueeze(0)],dim=1)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='1')
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--pth', type=str, default='31')
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--ckpt', type=str, default='/home/eunu/vid_stab/ckpt/ckpt_w5_s10')
    parser.add_argument('--save_path', type=str, default='/home/eunu/vid_stab/ckpt/test1')
    parser.add_argument('--h5_path', type=str, default='/hdd/DeepStab/h5/08.h5')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    test(args)