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

def video_to_frames(filepath):
    cap = cv2.VideoCapture(f'{filepath}.avi')
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imwrite(f'{filepath}/{str(n).zfill(4)}.jpg', frame)
        print(f'Saved {filepath}/{str(n).zfill(4)}.jpg')
        n += 1


if __name__ == '__main__':
    path = '/hdd/DeepStab/'
    for folder, filename in walk(path):
        name, ext = os.path.splitext(filename)
        if not is_video(ext): continue
        dir = os.path.join(folder,name)
        print(dir)
        try: os.mkdir(dir)
        except: pass
        video_to_frames(dir)