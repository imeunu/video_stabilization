import os

import cv2


'''
Divide the video into frames
'''

def walk(folder):
    for dirpath, dirs, files in os.walk(folder):
        for filename in files:
            yield dirpath, filename

def is_video(ext):
    if ext == '.avi': return True
    else: return False

def video_to_frames(vid_path, save_path):
    cap = cv2.VideoCapture(f'{vid_path}.avi')
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imwrite(f'{save_path}/{str(n).zfill(4)}.jpg', frame)
        print(f'Saved {save_path}/{str(n).zfill(4)}.jpg')
        n += 1


if __name__ == '__main__':
    path = '/Volumes/HD/imeunu/DeepStab/DeepStab/unstable'
    save = '/Users/eunu/Desktop/unstable'
    for folder, filename in walk(path):
        name, ext = os.path.splitext(filename)
        if not is_video(ext): continue
        vid_path = os.path.join(folder,name)
        save_path = os.path.join(save, name)
        print(save_path)
        try: os.mkdir(save_path)
        except: pass
        video_to_frames(vid_path, save_path)