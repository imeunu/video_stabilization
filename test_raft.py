import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small
def tensor_from_path(path):
    img = Image.open(path)
    img = np.array(img)
    img = img / 255
    img = torch.Tensor(img).permute(2,0,1)
    return img.unsqueeze(0)
class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]
    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
class Padder:
    def __init__(self, dims):
        self.ht, self.wd = dims[-2:]



if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= '2'
    i1 = '/home/eunu/vid_stab/opticalflow/demo-frames/frame_0016.png'
    i2 = '/home/eunu/vid_stab/opticalflow/demo-frames/frame_0017.png'
    
    i1, i2 = tensor_from_path(i1), tensor_from_path(i2)
    padder = InputPadder(i1.size())
    i1, i2 = padder.unpad(i1), padder.unpad(i2)
    print(i1.size(), i2.size()); sys.exit()

    device = torch.device('cuda')
    raft = raft_small(pretrained=True)
    raft = raft.eval().to(device)

    i1, i2 = i1.to(device), i2.to(device)
    out = raft(i1, i2)
    print(len(out))
    # last element of the output list!