import argparse
import os

import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import RNN
from rnn_loader import VideoDataset


def record_history(path, idx, loss):
    try:
        with open(path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow({idx, loss})
    except:
        f = open(path, 'w')
        f.write({idx, loss})
        f.close()

def load_each_model(args, ckpt, device):
    model = RNN(hidden_dim=args.hidden_dim)
    model = nn.DataParallel(model)
    model.to(device)
    ckpt_path = f'{args.ckpt_read}/{ckpt}'
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    return model.eval()

def measure(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = VideoDataset(args, train=False)
    testset = DataLoader(dataset, num_workers=2, batch_size=args.batch_size)
    criterion = torch.nn.MSELoss()
    csv_path, n = f'{args.ckpt_save}/measure.csv', 0
    while (n < args.epoch):
        for i, ckpt in enumerate(sorted(os.listdir(args.ckpt_read))):
            if i < n: continue
            try: model = load_each_model(args, ckpt, device)
            except: continue
            running_loss = 0; print(f'{n} Measuring {args.ckpt_read}/{ckpt}...')
            with torch.no_grad():
                with tqdm(total=len(testset)) as pbar:
                    for batch in testset:
                        inputx, target = batch
                        inputx, target, loss = inputx.cuda(), target.cuda(), 0
                        inputx = inputx.to(device=device).float()
                        target = target.to(device=device).float()
                        for j in range(inputx.size(1)):
                            if not j:
                                h = torch.zeros(inputx.size(0),20,18,32)
                            output, h = model(inputx[:,j,:,:,:], h)
                            loss += criterion(output, target[:,j,:,:,:])
                        running_loss += loss.item()
                        pbar.update(1)
                record_history(csv_path, i+1, running_loss/len(testset))
            print(f'name: {ckpt}, test loss: {running_loss/len(testset)}')
            n += 1

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--cuda', type=str, default='1')
    parser.add_argument('--ckpt_read', type=str, default='/home/eunu/vid_stab/ckpt/rnn12')
    parser.add_argument('--ckpt_save', type=str, default='/home/eunu/vid_stab/ckpt/rnn12')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=80)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--ssuuu', type=bool, default=False)
    parser.add_argument('--dataset', type=str, default='vr')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    measure(args)