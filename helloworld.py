import argparse
import os
import sys

import PIL
import torch
from torchvision import transforms

from opticalflow.core.raft import RAFT


def tensor_from_path(path):
    img = PIL.Image.open(path)
    tf = transforms.ToTensor()
    return tf(img)

def load_model(args, device):
    model = RAFT(args)
    model = torch.nn.DataParallel(model)
    ck_path = '/home/eunu/vid_stab/opticalflow/models/raft-kitti.pth'
    ck = torch.load(ck_path, map_location=device)
    model.load_state_dict(ck)
    return model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--small', type=bool, default=False)
    parser.add_argument('--mixed_precision', type=bool, default=True)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= '2'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args, device)
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    img1 = tensor_from_path('/home/eunu/vid_stab/opticalflow/demo-frames/frame_0016.png')
    img2 = tensor_from_path('/home/eunu/vid_stab/opticalflow/demo-frames/frame_0017.png')
    img1, img2 = img1.unsqueeze(0), img2.unsqueeze(0)
    out = model(img1,img2)