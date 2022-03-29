import argparse
import os

import cv2
import h5py
import numpy as np
from PIL import Image

'''
Create h5 file containing resized images.
All images are stored as numpy array.
'''

def get_img(path, args, aug):
    img = Image.open(path)
    img = img.resize(args.img_size, Image.BICUBIC)
    img = np.array(img)
    if aug == 'Hflip':
        img = cv2.flip(img, 0)
    elif aug == 'Vflip':
        img = cv2.flip(img, 1)
    elif aug == 'HVflip':
        img = cv2.flip(img, -1)
    return img

def frame_to_numpy_array(path, aug):
    arr = [get_img(os.path.join(path,name),aug) for name in os.listdir(path)]
    return np.array(arr)

def create_h5(dir, aug, args):
    stable_path = f'{args.root}/stable/{dir}'
    unstable_path = f'{args.root}/unstable/{dir}'

    stable = frame_to_numpy_array(stable_path, aug)
    unstable = frame_to_numpy_array(unstable_path, aug)
    
    h5 = h5py.File(f'{args.h5_path}/{dir.zfill(2).h5}', 'w')
    h5.create_dataset('stable', data=stable)
    h5.create_dataset('unstable', data=unstable)
    h5.create_dataset('frame', data=np.array(len(stable)))
    h5.close()

def generate(args):
    for path in sorted(os.listdir(f'{args.root}/stable')):
        if os.path.isdir(path):
            for aug in args.augmentation:
                create_h5(path, aug, args)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/hdd/DeepStab')
    parser.add_argument('--h5_path', type=str, default='/hdd/DeepStab/h5')
    parser.add_argument('--img_size', default=(320,180))
    parser.add_argument('--augmentation', default=(None, 'Hflip', 'Vflip', 'HVflip'))

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    try: os.mkdir(args.h5_path)
    except: pass
    generate(args)