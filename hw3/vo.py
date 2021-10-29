import open3d as o3d
import numpy as np
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp

class SimpleVO:
    def __init__(self, args):
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.K = camera_params['K']
        self.dist = camera_params['dist']
        
        self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, '*.png'))))

        self.px_ref = None
        self.px_cur = None
        self.cur_R = None
        self.cur_t = None
        self.new_cloud = None
        self.last_cloud = None

    def load_poses(self, R, t):
        axes = o3d.geometry.LineSet()
        w, h, z = 0.1, 0.1, 0.1
        axes.points = o3d.utility.Vector3dVector([[0, 0, 0], [w, h, z], [-w, h, z], [-w, -h, z], [w, -h, z]])
        axes.lines  = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]) # X, Y, Z
        axes.colors = o3d.utility.Vector3dVector([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])  # R, G, B
        axes.translate(t)
        axes.rotate(R)
        return axes

    def run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        queue = mp.Queue()
        p = mp.Process(target=self.process_frames, args=(queue, ))
        p.start()
        
        keep_running = True
        while keep_running:
            try:
                R, t = queue.get(block=False)
                if R is not None:
                    #TODO:
                    ## insert new camera pose here using vis.add_geometry()
                    axes = self.load_poses(R, t)
                    vis.add_geometry(axes)
                    
            except: pass
            
            keep_running = keep_running and vis.poll_events()
        vis.destroy_window()
        p.join()

    def featureMatching(self, img1, img2):
        ## Initiate ORB detector and BF matcher
        orb = cv.ORB_create()
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        ## Feature extraction
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        ## Match descriptors
        matches = bf.match(des1, des2)
        ## Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)[:100]
        
        self.px_ref = np.array([kp1[m.queryIdx].pt for m in matches])
        self.px_cur = np.array([kp2[m.trainIdx].pt for m in matches])

    def triangulatePoints(self, R, t):
        """Triangulates the feature correspondence points with
        the camera intrinsic matrix, rotation matrix, and translation vector.
        It creates projection matrices for the triangulation process."""

        # The canonical matrix (set as the origin)
        P0 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])
        P0 = self.K.dot(P0)
        # Rotated and translated using P0 as the reference point
        P1 = np.hstack((R, t))
        P1 = self.K.dot(P1)
        # Reshaped the point correspondence arrays to cv2.triangulatePoints's format
        point1 = self.px_ref.reshape(2, -1)
        point2 = self.px_cur.reshape(2, -1)

        return cv.triangulatePoints(P0, P1, point1, point2).reshape(-1, 4)[:, :3]

    def getScale(self):
        """ Returns the relative scale based on the 3-D point clouds
         produced by the triangulation_3D function. Using a pair of 3-D corresponding points
         the distance between them is calculated. This distance is then divided by the
         corresponding points' distance in another point cloud."""

        min_idx = min([self.new_cloud.shape[0], self.last_cloud.shape[0]])
        ratios = []  # List to obtain all the ratios of the distances
        for i in range(min_idx):
            if i > 0:
                Xk = self.new_cloud[i]
                p_Xk = self.new_cloud[i - 1]
                Xk_1 = self.last_cloud[i]
                p_Xk_1 = self.last_cloud[i - 1]

                if np.linalg.norm(p_Xk - Xk) != 0:
                    ratios.append(np.linalg.norm(p_Xk_1 - Xk_1) / np.linalg.norm(p_Xk - Xk))

        d_ratio = np.median(ratios) # Take the median of ratios list as the final ratio
        return d_ratio

    def process_frames(self, queue):
        ## Initiate ORB detector
        orb = cv.ORB_create()

        ## Process first frame
        R, t = np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
        queue.put((R, t))
        cur_img = cv.imread(self.frame_paths[0])

        ### Draw feature points on img 
        kp, des = orb.detectAndCompute(cur_img, None)
        im_with_keypoints = cv.drawKeypoints(cur_img, kp, np.array([]), (0,255,0))
        cv.imshow('frame', im_with_keypoints)     

        ## Process second frame
        prev_img = cur_img
        cur_img = cv.imread(self.frame_paths[1])

        ### Draw feature points on img 
        kp, des = orb.detectAndCompute(cur_img, None)
        im_with_keypoints = cv.drawKeypoints(cur_img, kp, np.array([]), (0,255,0))
        cv.imshow('frame', im_with_keypoints)

        self.featureMatching(prev_img, cur_img)

        ### Find essential matrix from the computed correspondences
        E, mask = cv.findEssentialMat(self.px_cur, self.px_ref, self.K)
        ### Recover camera pose from the essential matrix
        retval, self.cur_R, self.cur_t, mask = cv.recoverPose(E, self.px_cur, self.px_ref, self.K)
        queue.put((self.cur_R, self.cur_R.dot(self.cur_t)))
        ### Triangulation, returns 3-D point cloud
        self.new_cloud = self.triangulatePoints(self.cur_R, self.cur_t)
        ### The new frame becomes the previous frame
        self.last_cloud = self.new_cloud
        

        ## Process default frames
        for frame_path in self.frame_paths[2:]:

            prev_img = cur_img
            cur_img = cv.imread(frame_path)
            self.featureMatching(prev_img, cur_img)

            ### Find essential matrix from the computed correspondences
            E, mask = cv.findEssentialMat(self.px_cur, self.px_ref, self.K)
            
            ### Recover camera pose from the essential matrix
            retval, rel_R, rel_t, mask = cv.recoverPose(E, self.px_cur, self.px_ref, self.K,)

            ### Triangulation, returns 3-D point cloud
            self.new_cloud = self.triangulatePoints(rel_R, rel_t)

            ### Scaling the trajectory
            scale = self.getScale()

            if rel_t[2] > rel_t[0] and rel_t[2] > rel_t[1]:  # Accepts only dominant forward motion
                self.cur_t = self.cur_t + self.cur_R.dot(rel_t)
                self.cur_R = rel_R.dot(self.cur_R)
                queue.put((self.cur_R, self.cur_t))

            ### The new frame becomes the previous frame
            self.last_cloud = self.new_cloud

            ### Draw feature points on img
            kp, des = orb.detectAndCompute(cur_img, None)
            im_with_keypoints = cv.drawKeypoints(cur_img, kp, np.array([]), (0,255,0))
            cv.imshow('frame', im_with_keypoints)
            if cv.waitKey(30) == 27: break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='camera_parameters.npy', help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()
