import os

import cv2
import numpy as np


def get_toy():
    '''Assign frame pairs to variables'''
    stable, unstable, root = [], [], '/Users/eunu/Desktop/code/video_stabilization/toy/consective'
    # adjust the root variable with your environment
    for i in ['stable', 'unstable']:
        for j in os.listdir(f'{root}/{i}'):
            img = cv2.imread(f'{root}/{i}/{j}')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if i == 'stable':
                stable.append(img)
            else:
                unstable.append(img)
    return stable, unstable

def find_H(f1, f2):
    '''Find homography transformation of two images'''
    # Find feature descriptor with SIFT.
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(f1, None)
    kp2, des2 = sift.detectAndCompute(f2, None)

    # Match the features.
    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)
    sorted_matches = sorted(matches, key = lambda x: x.distance)
    res = cv2.drawMatches(f1, kp1, f2, kp2, sorted_matches[:30], None, flags = 2)

    # Remove outliers and find Homography. You can adjust hyperparameters.
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape((-1, 1, 2))
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape((-1, 1, 2))
    H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return H

def find_t(f1, f2):
    '''Extract affine translation parameters from H'''
    H = find_H(f1, f2)
    return (H[0][2], H[1][2])

def find_camera_distance(stable, unstable):
    '''Find x,y distance between camera center'''
    t = [find_t(stable[i], unstable[i]) for i in range(len(stable))]
    return np.mean(t, axis=0)

def decompose_H(H):
    '''Decompose homography matrix by SVD and find matrices R, t'''
    u, s, vh = np.linalg.svd(np.dot(H,H.transpose()))
    v = (u + vh) / 2
    l1, l2, l3 = s
    l1, l2, l3 = sorted(np.sqrt((l1, l2, l3)), reverse=True)
    z1, z3 = get_z(l1, l3)
    return z1, z3

def get_z(l1, l2):
    root = np.sqrt(1 + 4 * l1 * l2 / (l1 - l2) ** 2)
    z1, z2 = (-1 + root) / (2 * l1 * l2), (-1 - root) / (2 * l1 * l2)
    return z1, z2

if __name__ == '__main__':
    stable, unstable = get_toy()
    # print(find_camera_distance(stable, unstable))
    h = find_H(stable[0], unstable[0])
    print(decompose_H(h))