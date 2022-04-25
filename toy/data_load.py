import os

import cv2
import numpy as np

from torch.utils.data import Dataset


def walk(folder):
    for dirpath, dirs, files in os.walk(folder):
        for filename in files:
            yield dirpath, filename

def generate(unstable_dir, stable_dir):
    unstable_videos = os.listdir(unstable_dir)
    stable_videos = os.listdir(stable_dir)
    if not unstable_videos:
        raise RuntimeError(f'No input file found in {unstable_dir}')
    elif not stable_videos:
        raise RuntimeError(f'No input file found in {stable_dir}')
    elif len(unstable_videos) != len(stable_videos):
        raise RuntimeError(f"Numbers of file don't match")
    
    # stable = []; count = 0
    # for folder, filename in walk(stable_dir):
    #     path = os.path.join(folder, filename)
    #     cap = cv2.VideoCapture(path)
    #     video = []
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret: break
    #         video.append(frame)
    #     video = np.array(video)
    #     for i in range(len(video)):
    #         stable.append(video[i])
    #     print(f'loaded {filename}')
    #     count+=1
    #     if count>10:
    #         break
    # np.save('stable.npy',stable)

    unstable = []; count=0
    for folder, filename in walk(unstable_dir):
        path = os.path.join(folder, filename)
        cap = cv2.VideoCapture(path)
        video = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            video.append(frame)
        video = np.array(video)
        for i in range(len(video)):
            unstable.append(video[i])
        print(f'loaded {filename}')
        count+=1
        if count>10:
            break
    np.save('unstable.npy',unstable)
    

if __name__ == '__main__':
    path = 'E:\\imeunu\\DeepStab\\DeepStab'
    unstable = os.path.join(path,'unstable')
    stable = os.path.join(path,'stable')
    generate(unstable,stable)