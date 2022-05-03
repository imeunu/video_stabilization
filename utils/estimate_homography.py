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

def mat_minor(M, m, n):
    M = np.delete(M, m, 0)
    M = np.delete(M, n, 1)
    return (np.linalg.det(M))

def sign(M, m, n):
    return (np.sign(mat_minor(M, m, n)))

def normalize(vector):
    norm = np.linalg.norm(vector)
    return (vector / norm)

def compute_v(S):
    return np.sqrt(2 * (1 + np.trace(S)) ** 2 + 1 - np.trace(S*S))

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

    # Scale the matrix with median eigenvalue
    # _, v, _ = np.linalg.svd(H)
    # H = H / np.median(v)
    return (H)

def compute_Rtn(H):
    num , R, t, _ = cv2.decomposeHomographyMat(H)
    for i in range(num):
        if R[i][2][2] < 0: continue
        if t[i][0]: continue
        return (R[i], t[i])
    return (None)

'''
def choose_validZ(S):
    za1 = S[0][1] + np.sqrt(mat_minor(S,2,2)) / S[1][1]
    zb1 = S[0][1] - np.sqrt(mat_minor(S,2,2)) / S[1][1]
    za3 = (S[1][2] - sign(S,0,0) * np.sqrt(S,0,0)) / S[1][1]
    zb3 = (S[1][2] + sign(S,0,0) * np.sqrt(S,0,0)) / S[1][1]
    aa = 1 + za1 ** 2 + za3 ** 2
    ab = 1 + zb1 ** 2 + zb3 ** 2
    b = 2 + np.trace(S)
    v = np.sqrt(2 * (1 + np.trace(S) ** 2) + 1 - np.trace(S * S))
    wa = (b - v) / aa
    wb = (b - v) / ab
    ya = np.sqrt(wa) * np.array([za1,1,za3])
    yb = np.sqrt(wb) * np.array([zb1,1,zb3])

def compute_Rtn(H):
    S = np.matmul(H, H.transpose()) - np.identity(3)
    if not S[1][1] or S[2][2]:
        raise Exception('s22 and s33 must be different from 0')
    choose_validZ(S)
    # p22 (eq.68)
    return 

    v = compute_v(S)
    t_norm = 2 + np.trace(S) - v
    mode = np.argmax(abs(S.diagonal()))

    ta, tb = t_from_S(S, mode, t_norm)

    rho = np.sqrt(2 + np.trace(S) + compute_v(S))
    n_scale = sign(S, mode, mode) * rho / (2 * t_norm)
    na, nb = n_scale * (tb - ta), n_scale * (ta - tb)

    Ra, Rb = H - np.outer(ta, na), H - np.outer(tb, nb)
    return

def t_from_S(S, mode, t_norm):
    if mode == 0:
        ta = [S[0][0], S[0][1] + np.sqrt(mat_minor(S, 2, 2)),
                S[0][2] + sign(S, 1, 2) * mat_minor(S, 1, 1)]
        tb = [S[0][0], S[0][1] - np.sqrt(mat_minor(S, 2, 2)),
                S[0][2] + sign(S, 1, 2) * mat_minor(S, 1, 1)]
    elif mode == 1:
        ta = [S[0][1] + np.sqrt(mat_minor(S, 2, 2)), S[1][1],
                S[1][2] - sign(S, 0, 2) * mat_minor(S, 0, 0)]
        tb = [S[0][1] - np.sqrt(mat_minor(S, 2, 2)), S[1][1],
                S[1][2] + sign(S, 0, 2) * mat_minor(S, 0, 0)]
    elif mode == 2:
        ta = [S[0][2] + sign(S, 0, 2) * np.sqrt(mat_minor(S, 1, 1)),
                S[1][2] + np.sqrt(S, 0, 0), S[2][2]]
        tb = [S[0][2] - sign(S, 0, 2) * np.sqrt(mat_minor(S, 1, 1)),
                S[1][2] - np.sqrt(S, 0, 0), S[2][2]]
    
    ta, tb = normalize(ta), normalize(tb)
    ta, tb = t_norm * ta, t_norm * tb
    return (ta, tb)
'''

# def find_t(f1, f2):
#     '''Extract affine translation parameters from H'''
#     H = find_H(f1, f2)
#     return (H[0][2], H[1][2])

# def find_camera_distance(stable, unstable):
#     '''Find x,y distance between camera center'''
#     t = [find_t(stable[i], unstable[i]) for i in range(len(stable))]
#     return (np.mean(t, axis=0))

# def decompose_H(H):
#     '''Decompose homography matrix by SVD and find matrices R, t'''
#     u, s, vh = np.linalg.svd(np.dot(H,H.transpose()))
#     v1, _, v3 = (u + vh) / 2
#     l1, _, l3 = sorted(np.sqrt(s), reverse=True)
#     z1, z3 = get_z(l1, l3)
#     v_1, v_3 = get_v(z1, v1, l1, l3), get_v(z3, v3, l1, l3)
#     t, n = get_t_n(v_1, v_3, z1, z3)
#     Itnt = np.identity(3) + np.outer(t.transpose(), n)
#     R = np.matmul(H, np.linalg.inv(Itnt))
#     return (R, t)

# def get_z(l1, l2):
#     root = np.sqrt(1 + 4 * l1 * l2 / (l1 - l2) ** 2)
#     z1, z2 = (-1 + root) / (2 * l1 * l2), (-1 - root) / (2 * l1 * l2)
#     return (z1, z2)

# def get_v(zeta, v, l1, l2):
#     v_norm = (zeta ** 2) * ((l1 - l2) ** 2) + 2 * zeta * (l1 * l2 - 1) + 1
#     norm = np.sqrt(v_norm)
#     return (norm * v)

# def get_t_n(v_1, v_3, z1, z3):
#     # t, n can be either positive or negative
#     # Also, it can be v_1 - v_3
#     t = v_1 + v_3 / (z1 - z3)
#     n = (z1 * v_3 + z3 * v_1) / (z1 - z3)
#     return (t, n)

if __name__ == '__main__':
    # print(find_camera_distance(stable, unstable))
    for i in range(4):
        stable, unstable = get_toy(i)
        h = find_H(stable, unstable)
        R, t = compute_Rtn(h)