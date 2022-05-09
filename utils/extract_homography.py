import os

import csv
import cv2

import estimate_homography as homography


def save_csv(parameters, path):
    with open(path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(parameters)

def arange_parameters(name, angles, translations):
    alpha, beta, gamma = angles
    tx, ty, tz = translations
    return (name, alpha, beta, gamma, tx.item(), ty.item(), tz.item())

def extract_parameters(root, vid_num):
    csv_path = '/Users/eunu/Desktop/code/video_stabilization/parameters.csv'
    unstable_path = '/Users/eunu/Desktop/unstable'
    save_csv([vid_num], csv_path)
    for i in os.listdir(f'{root}/stable/{vid_num}'):
        stable = cv2.imread(f'{root}/stable/{vid_num}/{i}')
        stable = cv2.cvtColor(stable, cv2.COLOR_BGR2RGB)
        # unstable = cv2.imread(f'{root}/unstable/{vid_num}/{i}')
        unstable = cv2.imread(f'{unstable_path}/{vid_num}/{i}')
        unstable = cv2.cvtColor(unstable, cv2.COLOR_BGR2RGB)
        H = homography.find_H(stable, unstable)
        try:
            R, t = homography.compute_Rt(H)
            angles = homography.decompose_R(R)
            parameters = arange_parameters(i, angles, t)
            save_csv(parameters, csv_path)
            print(parameters)
        except: continue
    print(f'Saved video number {vid_num}')

if __name__ == '__main__':
    root = '/Volumes/HD/imeunu/DeepStab/DeepStab/'
    for vid_num in range(1,62):
        extract_parameters(root, vid_num)