import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, args, train=True):
        super().__init__()
        self.seq_len = args.seq_len
        if args.dataset == 'ds':
            h5_path = '/hdd/DeepStab/h5'
            if not train: dataset = sorted(os.listdir(h5_path))[-8:]
            else: dataset = sorted(os.listdir(h5_path))[:-8]
        elif args.dataset == 'vr':
            h5_path = '/home/eunu/VR/h5'
            if not train: dataset = sorted(os.listdir(h5_path))[-1:]
            elif train == 'all': dataset = sorted(os.listdir(h5_path))
            else: dataset = sorted(os.listdir(h5_path))[:-1]
        if not args.augmentation:
            dataset = sorted(data for data in dataset if 'None' in data)

        stable, unstable, frame_list = [], [], []
        for filename in dataset:
            with h5py.File(f'{h5_path}/{filename}', 'r') as f:
                stable.append(np.array(f['stable']))
                unstable.append(np.array(f['unstable']))
                frame_list.append(np.array(f['frame']))
        
        self.stable = stable
        self.unstable = unstable
        self.frame_list = np.array(frame_list) - self.seq_len + 1
    
    def __len__(self):
        return sum(self.frame_list)
    
    def __getitem__(self, idx):
        vid_idx = 0
        while (int(idx - self.frame_list[vid_idx]) >= 0):
            idx -= int(self.frame_list[vid_idx])
            vid_idx += 1
        return self.get_item(vid_idx, idx)
    
    def get_item(self, vid_idx, idx):
        x = self.to_tensor(self.unstable[vid_idx][idx : idx+self.seq_len])
        # size of x: (batch_size, seq_len, ch, ht, wd)

        target = self.to_tensor(self.stable[vid_idx][idx : idx+self.seq_len])
        # size of target: (batch_size, seq_len, ch, ht, wd)
        
        return (x, target)
    
    @classmethod
    def to_tensor(cls, img):
        return torch.Tensor(img / 255).permute(0,3,1,2)