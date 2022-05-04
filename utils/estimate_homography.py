import os

import cv2
import numpy as np


def get_toy(n=0):
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
        img = cv2.imread(f'{root}/{i}/000{n}.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if i == 'stable': stable = img
        else: unstable = img
    return (stable, unstable)

def isRotationMatrix(R, error=1e-6):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return (R < error)

def find_H(f1, f2, scale=False):
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

    # Scale the matrix with median eigenvalue.
    if scale:
        _, v, _ = np.linalg.svd(H)
        H = H / np.median(v)
    return (H)

def compute_Rt(H):
    num , R, t, _ = cv2.decomposeHomographyMat(H, np.eye(3))
    for i in range(num):
        if R[i][0,0] < 0: continue
        if t[i][0] < 0: continue
        if isRotationMatrix(R[i]): return (R[i], t[i])
    return (None)

def decompose_R(R, error=1e-6):
    sy = np.sqrt(R[0,0] ** 2 + R[1,0] ** 2)
    singular = (sy < error)
    x = np.arctan2(R[2,1], R[2,2])
    y = np.arctan2(-R[2,0], sy)
    z = np.arctan2(R[1,0], R[0,0]) if not singular else 0
    return (x, y, z)

if __name__ == '__main__':
    stable, unstable = get_toy()
    h = find_H(stable, unstable)
    R, t = compute_Rt(h)
    tx, ty, _ = t
    alpha, beta, gamma = decompose_R(R)