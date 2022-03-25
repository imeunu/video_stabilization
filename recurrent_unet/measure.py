import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from UNet.unet_model import UNet
from utils.testloader import TestSet

def load_model(args, filename, device):
    model = UNet(15,3)
    ckpt_path = f'{args.ckpt_path}/{filename}'
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model = torch.nn.DataParallel(model)
    return model

def measure(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TestSet(args)
    testset = DataLoader(dataset, num_workers=4)
    criterion = torch.nn.MSELoss()

    for i, ckpt in enumerate(sorted(os.listdir(args.ckpt_path),reverse=True)):
        try: model = load_model(args, ckpt, device)
        except: continue
        running_loss = 0
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
        print(f'name: {ckpt}, test loss: {running_loss}')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='1')
    parser.add_argument('--ckpt_path', type=str, default='/home/eunu/vid_stab/ckpt/recurrent_unet_5_10')
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--ssuuu', type=bool, default=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    measure(args)