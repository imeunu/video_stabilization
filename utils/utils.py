import csv

import torch.nn as nn
from torchvision.models.optical_flow import raft_small as raft


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

def record_history(path, idx, loss):
    try:
        with open(path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow({idx, loss})
    except:
        f = open(path, 'w')
        f.write({idx, loss})
        f.close()

def load_optical_flow(device):
    model = raft(pretrained=True)
    model = nn.DataParallel(model)
    model.to(device)
    return model.eval()