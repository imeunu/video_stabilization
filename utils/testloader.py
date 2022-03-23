import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TestSet(Dataset):
    def __init__(self, args):
        super().__init__()
        h5_path = '/hdd/DeepStab/h5'
        self.window = args.window
        self.seq_len = args.seq_len
        self.ssuuu = args.ssuuu
        
        stable, unstable, frame_list = [], [], []
        for f in os.listdir(h5_path):
            if f not in ['08.h5', '18.h5', '22.h5']: continue
            with h5py.File(f'{h5_path}/{f}', 'r') as f:
                stable.append(np.array(f['stable']))
                unstable.append(np.array(f['unstable']))
                frame_list.append(np.array(f['frame']))
        
        self.stable = stable
        self.unstable = unstable
        self.frame_list = np.array(frame_list) - self.seq_len + 1
    
    def __len__(self):
        return sum(self.frame_list) - self.window * len(self.frame_list)
    
    def __getitem__(self, idx):
        vid_idx = 0
        while (int(idx - self.frame_list[vid_idx]) >= 0):
            idx -= int(self.frame_list[vid_idx])
            vid_idx += 1
        return self.get_item(vid_idx, idx)
    
    def get_item(self, vid_idx, idx):
        n = self.window // 2
        if self.ssuuu:
            s = self.to_tensor(
                    self.stable[vid_idx][idx : idx+n]
            ) # s: size (2,3,72,128)
            # s = torch.cat(torch.tensor_split(s, s.size(0)), dim=1) # s: size (1,6,72,128)
            u = self.to_tensor(
                    self.unstable[vid_idx][idx+n : idx+self.seq_len]
            ) # u: size (18,3,72,128)
            # u = torch.cat(torch.tensor_split(u, u.size(0)), dim=1) # u: size (1,54,72,128)
            x = torch.cat([s,u], dim=0)
        else:
            x = self.to_tensor(self.unstable[vid_idx][idx : idx+self.seq_len])
            x = torch.cat(torch.tensor_split(x, x.size(0)), dim=1)
            # x: size (1,60,72,128)

        target = self.to_tensor(
            self.stable[vid_idx][idx+n : idx+self.seq_len-n]
        ) # target: size (16,3,72,128)
        
        return (x, target)
    
    @classmethod
    def to_tensor(cls, img):
        img = torch.tensor(img / 255)
        img = img.permute(0,3,1,2).contiguous()
        return img