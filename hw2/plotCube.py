import open3d as o3d
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys, os
import pandas as pd

def get_transform_mat(rotation, translation, scale):
    r_mat = R.from_quat(rotation).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat.squeeze(), translation.reshape(3, 1)], axis=1)
    return transform_mat

def transform(vertice, inMatrix, exMatrix):
    pos = inMatrix.dot(exMatrix.dot(np.append(vertice, 1.0)))
    pos /= pos[-1]
    return [int(pos[0]), int(pos[1])]

def get_2d_points(pts, v0, v1, v2, v3, inMatrix, exMatrix, surfaceID):
    numPoints = 8
    head = []

    vec0 = v2 - v0
    for i in range(numPoints):
        head.append(v0 + vec0 * i / (numPoints-1))

    vec1 = v1 - v0
    for i in range(numPoints):
        for j in range(numPoints):
            pt = head[i] + vec1 * j  / (numPoints-1)
            pts.append((transform(pt, inMatrix, exMatrix), pt[-1], surfaceID))


if __name__ == '__main__':
    
    ## load cude vertices
    
    vertices  = np.load('cube_vertices.npy')

    ## Initialize
    images_df = pd.read_pickle("data/images.pkl")
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    numSurface = 6
    colorSurface = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]

    ## covert frames to video
    path_dir = './data/frames'
    path_out  = './video.mp4'
    frames = []
    files = [f for f in os.listdir(path_dir) if f.find('valid') != -1]
    files.sort(key=lambda x: int(x[9:-4]))
    
    for f in files:
        filename = os.path.join(path_dir, f)
        img = cv2.imread(filename)
        h, w, ch = img.shape
        size = (w, h)

        ## plot cude in img
        ### Get camera pose groudtruth (extrinsic parameter)
        ground_truth = images_df.loc[images_df["NAME"] == f]
        rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
        tvec_gt = ground_truth[["TX","TY","TZ"]].values
        extrinsicMatrix = get_transform_mat(rotq_gt, tvec_gt, 1)

        ### compute the coordinate on image plane
        pts = []
        get_2d_points(pts, vertices[0], vertices[1], vertices[2], vertices[3], cameraMatrix, extrinsicMatrix, 0)
        get_2d_points(pts, vertices[4], vertices[5], vertices[6], vertices[7], cameraMatrix, extrinsicMatrix, 1)
        get_2d_points(pts, vertices[0], vertices[1], vertices[4], vertices[5], cameraMatrix, extrinsicMatrix, 2)
        get_2d_points(pts, vertices[2], vertices[3], vertices[6], vertices[7], cameraMatrix, extrinsicMatrix, 3)
        get_2d_points(pts, vertices[0], vertices[2], vertices[4], vertices[6], cameraMatrix, extrinsicMatrix, 4)
        get_2d_points(pts, vertices[1], vertices[3], vertices[5], vertices[7], cameraMatrix, extrinsicMatrix, 5)

        pts.sort(key=lambda x: -x[1])

        for (pt, depth, surfaceID) in pts:
            # if pt[0] >= 0 and pt[0] < h and pt[1] >= 0 and pt[1] < w:
            cv2.circle(img, tuple(pt), 2, colorSurface[surfaceID], 5)

        frames.append(img)

    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc('m','p','4','v'), 10, size)

    for frame in frames:
        # writing to a image array
        out.write(frame)
    out.release()

    
