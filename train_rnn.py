import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rnn.model import RNN
from rnn.rnn_loader import VideoDataset
from utils import utils


def load_model(args, device):
    model = RNN(hidden_dim=args.hidden_dim)
    model = nn.DataParallel(model)
    ckpt_path = f'{args.ckpt_read}/{str(args.pth).zfill(3)}.pth'
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    return model

def train(args):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try: os.mkdir(args.ckpt_save)
    except: pass
    trainset = VideoDataset(args, train='all')
    trainset = DataLoader(trainset, shuffle=True, batch_size=args.batch_size,
                        num_workers=8, pin_memory=True)
    csv_path = f'{args.ckpt_save}/history.csv'
    # writer = SummaryWriter('runs/DeepStab')
    model = RNN(hidden_dim=args.hidden_dim)
    model.to(device)
    model = nn.DataParallel(model)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    torch.save(optimizer.state_dict(), f'{args.ckpt_save}/opt.pth')

    if args.re_train:
        model = load_model(args, device)
        # for _ in range(args.pth):
        #     scheduler.step()
    else:
        args.pth = 0
    criterion = nn.MSELoss()
    
    if args.flow_loss:
        optflow = nn.DataParallel(utils.load_optical_flow(device))
        flow_target_size = (args.batch_size, 2) + args.img_size
        # flow_target = torch.zeros(flow_target_size)
        padder = utils.InputPadder(flow_target_size)
        # flow_target = padder.unpad(flow_target)
        # flow_target = flow_target.to(device)

    for epoch in range(args.pth, args.epoch):
        running_loss = 0
        with tqdm(total=len(trainset), desc=f'Epoch{epoch+1}/lr{scheduler.get_last_lr()}') as pbar:
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
                    loss += args.lambda_flow * criterion(output, target[:,j,:,:,:])
                    if args.flow_loss and not j: # flow_loss and j=0
                        prev = padder.unpad(output)
                    elif args.flow_loss and j:
                        output = padder.unpad(output)
                        flow = optflow(prev, output)
                        flow_target = optflow(padder.unpad(target[:,j-1,:,:,:]),
                                            padder.unpad(target[:,j,:,:,:])) # Use GT flow
                        loss += criterion(flow[-1], flow_target[-1])
                        prev = output
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                running_loss += loss.item()
                print(f'Step: {i+1}, Running Loss: {loss.item()}')
                pbar.update(1)
        utils.record_history(csv_path, epoch+1, running_loss/len(trainset))
        # writer.add_scalar('training loss', running_loss/len(trainset))
        print(f'Epoch: {epoch+1}, Loss: {running_loss/len(trainset)}')
        scheduler.step()
        torch.save(model.state_dict(), args.ckpt_save + f'/{str(epoch+1).zfill(3)}.pth')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--ckpt_read', type=str, default='/home/eunu/vidstab/ckpt/finetune')
    parser.add_argument('--ckpt_save', type=str, default='/home/eunu/vidstab/ckpt/flow_gt')
    parser.add_argument('--hidden_dim', type=int, default=80)
    parser.add_argument('--img_size', default=(180,320))
    parser.add_argument('--dataset', type=str, default='vr')

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--augmentation', type=bool, default=False)
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--lambda_flow', type=float, default=0.05)
    parser.add_argument('--flow_loss', type=bool, default=True)

    parser.add_argument('--pth', type=int, default=72)
    parser.add_argument('--re_train', type=bool, default=True)
    parser.add_argument('--cuda', type=str, default='1')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    train(args)

    # started to add flow loss from 54th epoch