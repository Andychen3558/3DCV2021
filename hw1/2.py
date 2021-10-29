import sys
import numpy as np
import cv2 as cv
import time
api = __import__('1')
# from '1' import solve_homography

WINDOW_NAME = 'window'


def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        param[0].append([x, y])  

def backward_warping(img, dst, transformed_size):
    """
    :return: image after unwarping
    """
    new_h, new_w = transformed_size
    output = np.zeros((new_h, new_w, 3))
    dst = np.array([[c[1], c[0]] for c in dst])

    ## solve homography matrix
    src = np.array([[0, 0], [0, new_w-1], [new_h-1, 0], [new_h-1, new_w-1]])
    H = api.solve_homography(src, dst)

    ## do image warping
    height, width = np.linspace(0, new_h-1, new_h), np.linspace(0, new_w-1, new_w)
    iv, jv = np.meshgrid(height, width, indexing='ij')
    iv, jv = iv.flatten(), jv.flatten()
    coor = np.vstack((iv, jv, np.ones((1, iv.shape[0]))))
    out = H.dot(coor)
    out[:2] /= out[2]
    
    ## compute surrounding coordinates
    i, j = np.copy(out[0]), np.copy(out[1])
    i, j = np.clip(i, 0, img.shape[0]-1), np.clip(j, 0, img.shape[1]-1)
    i_min, j_min = np.floor(np.copy(i)).astype(int), np.floor(np.copy(j)).astype(int)
    i_max, j_max = np.floor(np.copy(i+1)).astype(int), np.floor(np.copy(j+1)).astype(int)

    i_min, j_min = np.clip(i_min, 0, img.shape[0]-1), np.clip(j_min, 0, img.shape[1]-1)
    i_max, j_max = np.clip(i_max, 0, img.shape[0]-1), np.clip(j_max, 0, img.shape[1]-1)
    
    ## do interpolation
    intensity_surr = np.zeros(shape=(i.shape[0], 4, 3))
    intensity_surr[:, 0, :] = img[i_min, j_min]
    intensity_surr[:, 1, :] = img[i_max, j_min]
    intensity_surr[:, 2, :] = img[i_min, j_max]
    intensity_surr[:, 3, :] = img[i_max, j_max]

    weight = np.zeros(shape=(i.shape[0], 4))
    weight[:, 0] = (i_max-i) * (j_max-j) / ((i_max-i_min) * (j_max-j_min))
    weight[:, 1] = (i-i_min) * (j_max-j) / ((i_max-i_min) * (j_max-j_min))
    weight[:, 2] = (i_max-i) * (j-j_min) / ((i_max-i_min) * (j_max-j_min))
    weight[:, 3] = (i-i_min) * (j-j_min) / ((i_max-i_min) * (j_max-j_min))
    weight = np.expand_dims(weight, axis=1)

    output = (weight @ intensity_surr).reshape(new_h, new_w, 3)
    return output
    


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[USAGE] python3 mouse_click_example.py [IMAGE PATH]')
        sys.exit(1)

    img = cv.imread(sys.argv[1])
    img = cv.resize(img, (img.shape[1]//3, img.shape[0]//3))

    points_add = []
    # cv.namedWindow(WINDOW_NAME)
    # cv.setMouseCallback(WINDOW_NAME, on_mouse, [points_add])
    # while True:
    #     img_ = img.copy()
    #     for i, p in enumerate(points_add):
    #         # draw points on img_
    #         cv.circle(img_, tuple(p), 2, (0, 255, 0), -1)
    #     cv.imshow(WINDOW_NAME, img_)

    #     key = cv.waitKey(20) % 0xFF
    #     if key == 27: break # exist when pressing ESC

    # cv.destroyAllWindows()
    
    # print('{} Points added'.format(len(points_add)))

    ts = time.time()
    points_add = [[112, 24], [509, 101], [3, 583], [402, 664]]

    ### apply backward warping with bilinear interpolation
    dst = np.array(points_add)
    output = backward_warping(img, dst, (img.shape[0], img.shape[1])).astype(np.uint8)

    cv.imshow('output', output)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # cv.imwrite('./images/part2.png', output)

    te = time.time()
    print('Elapse time: {}...'.format(te-ts))