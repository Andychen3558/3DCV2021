import open3d as o3d
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys, os
import pandas as pd

def load_point_cloud(points3D_df):

    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB'])/255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    return pcd

def load_axes():
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axes.lines  = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])          # X, Y, Z
    axes.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # R, G, B
    return axes

def load_poses(r, t):
    axes = o3d.geometry.LineSet()
    w, h, z = 0.25, 0.25, 0.25
    axes.points = o3d.utility.Vector3dVector([[0, 0, 0], [w, h, z], [-w, h, z], [-w, -h, z], [w, -h, z]])
    # axes.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axes.lines  = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]) # X, Y, Z
    axes.colors = o3d.utility.Vector3dVector([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])  # R, G, B
    axes.rotate(R.from_quat(r).as_matrix())
    axes.translate(t)
    return axes

def get_transform_mat(rotation, translation, scale):
    r_mat = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat, translation.reshape(3, 1)], axis=1)
    return transform_mat


# if len(sys.argv) != 2:
#     print('[Usage] python3 transform_cube.py /PATH/TO/points3D.txt')
#     sys.exit(1)

if __name__ == '__main__':
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    ## load point cloud
    points3D_df = pd.read_pickle("data/points3D.pkl")
    pcd = load_point_cloud(points3D_df)
    vis.add_geometry(pcd)

    ## load camera poses
    rotation = np.load('./Rotation.npy')
    translation = np.load('./Translation.npy')

    ## load axes
    for i in range(len(rotation)):
        axes = load_poses(rotation[i], translation[i])
        vis.add_geometry(axes)


    ## just set a proper initial camera view
    vc = vis.get_view_control()
    vc_cam = vc.convert_to_pinhole_camera_parameters()
    initial_cam = get_transform_mat(np.array([7.227, -16.950, -14.868]), np.array([-0.351, 1.036, 5.132]), 1)
    initial_cam = np.concatenate([initial_cam, np.zeros([1, 4])], 0)
    initial_cam[-1, -1] = 1.
    setattr(vc_cam, 'extrinsic', initial_cam)
    vc.convert_from_pinhole_camera_parameters(vc_cam)


    vis.run()
    vis.destroy_window()


