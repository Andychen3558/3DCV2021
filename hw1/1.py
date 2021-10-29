import sys
import numpy as np
import cv2 as cv
import random

def get_sift_correspondences(img1, img2):
    '''
    Input:
        img1: numpy array of the first image
        img2: numpy array of the second image

    Return:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    '''
    #sift = cv.xfeatures2d.SIFT_create()# opencv-python and opencv-contrib-python version == 3.4.2.16 or enable nonfree
    sift = cv.SIFT_create()             # opencv-python==4.5.1.48
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)
    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])
    
    # img_draw_match = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv.imshow('match', img_draw_match)
    # cv.waitKey(0)
    return points1, points2

def convert_to_homography_param(point_list):
    """
    :return: matrix of homography param (3 x N). N = width x height.
    """
    return np.vstack((point_list, np.ones((1, point_list.shape[1]))))

def normalize(point_list):
    # type: (np.ndarray) -> (np.ndarray, np.ndarray)
    """
    :param point_list: point list to be normalized
    :return: normalization results
    """
    m = np.mean(point_list[:2], axis=1)
    max_std = max(np.std(point_list[:2], axis=1)) + 1e-9
    c = np.diag([1 / max_std, 1 / max_std, 1])
    c[0][2] = -m[0] / max_std
    c[1][2] = -m[1] / max_std
    return np.dot(c, point_list), c

def solve_homography(u, v):
    N = u.shape[0]
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    A = np.zeros((2*N, 9))
    H = np.zeros((3, 3))

    dst = convert_to_homography_param(v.T)
    src = convert_to_homography_param(u.T)

    ## do normalization
    src, c1 = normalize(src)
    dst, c2 = normalize(dst)

    ## construct matrix A
    for i in range(N):
        A[2 * i] = np.array([0, 0, 0, -src[0][i], -src[1][i], -1, dst[1][i] * src[0][i], dst[1][i] * src[1][i], dst[1][i]])
        A[2 * i + 1] = np.array([src[0][i], src[1][i], 1, 0, 0, 0, -src[0][i] * dst[0][i], -src[1][i] * dst[0][i], -dst[0][i]])
    
    ## use svd to get homography matrix
    U, S, V_T = np.linalg.svd(A)
    H = V_T[-1].reshape(3, 3)
    H = np.dot(np.linalg.inv(c2), np.dot(H, c1))
    return H / H[2, 2]


if __name__ == '__main__':
    img1 = cv.imread(sys.argv[1])
    img2 = cv.imread(sys.argv[2])
    gt_correspondences = np.load(sys.argv[3])
    
    points1, points2 = get_sift_correspondences(img1, img2)
    

    ## random sample N correspondences to generate homography
    N = int(sys.argv[4])
    idx = random.sample(range(0, len(points1)), N)
    src, dst = points1[idx], points2[idx]
    # src, dst = points1[:N], points2[:N]

    ## solve homography matrix
    H = solve_homography(src, dst)
    
    ## use homography to compute error
    p_s = gt_correspondences[0]
    p_t = gt_correspondences[1]

    cnt = p_s.shape[0]
    err = 0.0
    for i in range(cnt):       
        trans = H.dot(np.append(p_s[i], 1.0))
        trans /= trans[-1]
        err += np.sqrt(np.sum(np.square(p_t[i] - trans[:2])))
    print("The reprojection error is %f" %(err/cnt))

