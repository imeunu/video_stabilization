import os

import cv2
import numpy as np

'''
Create a video with inferenced frame
'''

if __name__ == '__main__':
    path = '/home/eunu/vid_stab/ckpt/test1'
    video = []
    for i in range(len(os.listdir(path))):
        img_path = f'{path}/{str(i).zfill(4)}.jpg'
        img = cv2.imread(img_path)
        video.append(img)
        print(f'Opened {img_path}')
    video = np.array(video)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    output = cv2.VideoWriter('output.mp4', fourcc, 25, (720,1280))
    if not output.isOpened():
        print('File open failed!')
    for frame in video:
        output.write(frame)
    output.release()