import argparse
import os

import h5py
import numpy as np
from PIL import Image

'''
Create h5 file containing resized images.
All images are stored as numpy array.
'''

def image_from_path(path, args):
    img = Image.open(path)
    img = img.resize((128, 72), Image.BICUBIC)
    img = np.array(img)
    return img

def create_h5(dir, args):
    stable_path = f'{args.root}/stable/{dir}'
    unstable_path = f'{args.root}/unstable/{dir}'
    stable, unstable = [], []

    for img_path in os.list_dir(stable_path):
        for img_path in os.listdir(stable_path):
        img_path = os.path.join(stable_path, img_path)
        stable_frame = image_from_path(img_path, args)
        stable.append(stable_frame)
    for img_path in os.listdir(unstable_path):
        img_path = os.path.join(unstable_path, img_path)
        unstable_frame = image_from_path(img_path, args)
        unstable.append(unstable_frame)

    stable, unstable = np.array(stable), np.array(unstable)
    h5 = h5py.File(f'{args.h5_path}/{dir.zfill(2).h5}', 'w')
    h5.create_dataset('stable', data=stable)
    h5.create_dataset('unstable', data=unstable)
    h5.create_dataset('frame', data=np.array(len(stable)))
    h5.close()

def generate(args):
    os.chdir(f'{args.root}/stable')

    for path in sorted(os.listdir()):
        if os.path.isdir(path):
            create_h5(path, args)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/hdd/DeepStab')
    parser.add_argument('--h5_path', type=str, default='/hdd/DeepStab/h5')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    try: os.mkdir(args.h5_path)
    except: pass
    generate(args)