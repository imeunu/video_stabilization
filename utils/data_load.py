import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self,args):
        self.stable_dir = args.root + args.stable
        self.unstable_dir = args.root + args.unstable
        self.scale = args.scale
        vid_name, frame_list = [], []

        for i in range(1,59):
            vid_name.append(i)
            frame_list.append(len(os.listdir(os.path.join(
                                    self.unstable_dir, str(i)))))
        self.vids = vid_name
        if not self.vids:
            raise RuntimeError(f'No files in directory')
        self.frame_list = np.array(frame_list) - 4
    
    def __len__(self):
        return sum(self.frame_list) - 4 * len(self.vids)
    
    @classmethod
    def preprocess(cls, img, scale):
        w, h = img.size
        w_new, h_new = int(scale * w), int(scale * h)
        assert w_new > 0 and h_new > 0, 'Scale is too small'
        img = img.resize((w_new, h_new), resample = Image.BICUBIC)
        img = np.asarray(img) / 255
        return img.transpose((2,0,1))

    @classmethod
    def flat(cls,tensor):
        return tensor.shape(1,-1)
    
    @classmethod
    def load(cls, filename):
        return Image.open(filename)
    
    def consecutive_frames(self, folder, idx):
        img1 = self.load(os.path.join(self.unstable_dir, str(folder), str(idx) + '.png'))
        img1 = self.preprocess(img1, self.scale)
        img1 = torch.as_tensor(img1.copy()).float().contiguous()
        # img1 = self.flat(img1)

        img2 = self.load(os.path.join(self.unstable_dir, str(folder), str(idx + 1) + '.png'))
        img2 = self.preprocess(img2, self.scale)
        img2 = torch.as_tensor(img2.copy()).float().contiguous()
        # img2 = self.flat(img2)

        img3 = self.load(os.path.join(self.unstable_dir, str(folder), str(idx + 2) + '.png'))
        img3 = self.preprocess(img3, self.scale)
        img3 = torch.as_tensor(img3.copy()).float().contiguous()
        # img3 = self.flat(img3)

        img4 = self.load(os.path.join(self.unstable_dir, str(folder), str(idx + 3) + '.png'))
        img4 = self.preprocess(img4, self.scale)
        img4 = torch.as_tensor(img4.copy()).float().contiguous()
        # img4 = self.flat(img4)

        img5 = self.load(os.path.join(self.unstable_dir, str(folder), str(idx + 4) + '.png'))
        img5 = self.preprocess(img5, self.scale)
        img5 = torch.as_tensor(img5.copy()).float().contiguous()
        # img5 = self.flat(img5)

        target = self.load(os.path.join(self.stable_dir, str(folder), str(idx + 2) + '.png'))
        target = self.preprocess(target, self.scale)
        # target = self.flat(target)

        return (
            torch.cat([img1,img2,img3,img4,img5], dim=0),
            torch.as_tensor(target.copy()).float().contiguous()
        )
    
    def __getitem__(self, idx):
        vid_idx = 0
        while (idx - self.frame_list[vid_idx] >= 0):
            idx -= self.frame_list[vid_idx]; vid_idx += 1
        return self.consecutive_frames(vid_idx+1, idx)