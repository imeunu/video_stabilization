import os

import cv2
import numpy as np


def get_toy():
    '''Assign frame pairs to variables'''
    root = '/Users/eunu/Desktop/code/video_stabilization/toy/consecutive'
    # adjust the root variable with your environment
    for i in ['stable', 'unstable']:
        # for j in os.listdir(f'{root}/{i}'):
        #     img = cv2.imread(f'{root}/{i}/{j}')
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     if i == 'stable':
        #         stable.append(img)
        #     else:
        #         unstable.append(img)
        img = cv2.imread(f'{root}/{i}/0000.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if i == 'stable': stable = img
        else: unstable = img
    return (stable, unstable)

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
    res = cv2.drawMatches(f1,kp1,f2,kp2,sorted_matches[:30],None,flags=2)

    # Remove outliers and find Homography. You can adjust hyperparameters.
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape((-1,1,2))
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape((-1,1,2))
    H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return (H)

def find_t(f1, f2):
    '''Extract affine translation parameters from H'''
    H = find_H(f1, f2)
    return (H[0][2], H[1][2])

def find_camera_distance(stable, unstable):
    '''Find x,y distance between camera center'''
    t = [find_t(stable[i], unstable[i]) for i in range(len(stable))]
    return (np.mean(t, axis=0))

def decompose_H(H):
    '''Decompose homography matrix by SVD and find matrices R, t'''
    u, s, vh = np.linalg.svd(np.dot(H,H.transpose()))
    v1, _, v3 = (u + vh) / 2
    l1, _, l3 = sorted(np.sqrt(s), reverse=True)
    z1, z3 = get_z(l1, l3)
    v_1, v_3 = get_v(z1, v1, l1, l3), get_v(z3, v3, l1, l3)
    t, n = get_t_n(v_1, v_3, z1, z3)
    Itnt = np.identity(3) + np.outer(t.transpose(), n)
    R = np.matmul(H, np.linalg.inv(Itnt))
    return (R, t)

def get_z(l1, l2):
    root = np.sqrt(1 + 4 * l1 * l2 / (l1 - l2) ** 2)
    z1, z2 = (-1 + root) / (2 * l1 * l2), (-1 - root) / (2 * l1 * l2)
    return (z1, z2)

def get_v(zeta, v, l1, l2):
    v_norm = (zeta ** 2) * ((l1 - l2) ** 2) + 2 * zeta * (l1 * l2 - 1) + 1
    norm = np.sqrt(v_norm)
    return (norm * v)

def get_t_n(v_1, v_3, z1, z3):
    # t, n can be either positive or negative
    # Also, it can be v_1 - v_3
    t = v_1 + v_3 / (z1 - z3)
    n = (z1 * v_3 + z3 * v_1) / (z1 - z3)
    return (t, n)

def new_solution(H):
    S = np.matmul(H.transpose(), H)
    Ms = -mat_minor(S,1,1)
    return S, np.linalg.det(S)

def mat_minor(M,m,n):
    M = np.delete(M,m-1,0)
    M = np.delete(M,n-1,1)
    return (np.linalg.det(M))

def sign(x):
    if x >= 0: return 1
    else: return -1

if __name__ == '__main__':
    stable, unstable = get_toy()
    # print(find_camera_distance(stable, unstable))
    h = find_H(stable, unstable)
    print(h)
    print(new_solution(h))