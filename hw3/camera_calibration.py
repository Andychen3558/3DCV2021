
import numpy as np
import cv2 as cv
import glob, sys, argparse

class Calibrator:
    def __init__(self, args):
        self.args = args
        self.inner_w = args.w
        self.inner_h = args.h

        # termination criteria
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((self.inner_w * self.inner_h, 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.inner_h, 0:self.inner_w].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.
        self.imgs = []

    def run(self):
        self.video_reader = cv.VideoCapture(args.input)
        self.load_images()
        assert len(self.imgs) >= 4, print('=> Error: need a least 4 images to calibrate')

        ret, self.K, self.dist, self.rvec, self.tvecs = self.calibrate()
        print('=> Overall RMS re-projection error: {}'.format(ret))

        self.save_result()

        if args.show:
            self.show_result(self.imgs, self.imgpoints, self.K, self.dist, self.rvec, self.tvecs)


    def load_images(self):

        #for fname in images:
        print('=> press <q> to add image')
        print('=> press <space> to exit adding and start calibration')
        while True:
            #img = cv.imread(fname)
            ret, img = self.video_reader.read()
            if not ret: break
            #img = img[:, ::-1]
            cv.imshow('calibration video', img)
            key = cv.waitKey(10) #& 0xFF
            if key == ord(' '):
                self.imgs.append(img)
                print('save images: {}'.format(len(self.imgs)))
            if key == ord('q'):
                break
        cv.destroyAllWindows()

    def calibrate(self):

        print('=> start corner detection and calibration')

        for img in self.imgs:
            # Find the chess board corners
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, (self.inner_h, self.inner_w), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                self.objpoints.append(self.objp)
                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), self.criteria)
                self.imgpoints.append(corners2)
                # Draw and display the corners
                img_ = cv.UMat(img)
                cv.drawChessboardCorners(img_, (self.inner_h, self.inner_w), corners2, ret)
                cv.imshow('img', img_)
                cv.waitKey(45)

        cv.destroyAllWindows()
        
        assert len(self.objpoints) > 0, print('=> Error: no corner detected')
        assert len(self.objpoints) >= 4, print('=> Error: need a least 4 images to calibrate')
        print('=> corners found in {} images'.format(len(self.objpoints)))

        #ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        return cv.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)

    def save_result(self):
        print('Camera Intrinsic')
        print(self.K)
        print('Distortion Coefficients')
        print(self.dist)
        np.save(self.args.output, {'K': self.K, 'dist': self.dist})

    def show_result(self, imgs, imgpoints, mtx, dist, rvecs, tvecs):
        try:
            import open3d as o3d
        except Exception as e:
            print(e)
            return
        coord = np.eye(4)
        coord[1, 1] = -1
        coord[2, 2] = -1
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for i in range(self.inner_h+1):
            for j in range(self.inner_w+1):
                color = [0, 0, 0] if (i + j) % 2 == 0 else [0.9, 0.9, 0.9]
                mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=0.2)
                mesh.translate((i-1, - j, -0.2))
                mesh.paint_uniform_color(color)
                vis.add_geometry(mesh)
            
        def expand_batch(_m, _batch_size):
            b_m = np.repeat(np.expand_dims(_m, 0), _batch_size, axis=0)
            return b_m

        def create_camera(_img, _r_mat, _t_vec, _K):
            _h, _w = _img.shape[:2]
            verts = np.zeros((5, 3)).astype(np.float32)
            verts[1:3, 0] = _w
            verts[2:4, 1] = _h
            verts[:, 2] = 1.
            verts[4, 0] = _w / 2
            verts[4, 1] = _h / 2 
            verts = (expand_batch(np.linalg.inv(_K), 5) @ np.expand_dims(verts, -1))
            verts[4, -1] = 0.
            #verts = expand_batch(coord[:3, :3], 5) @ verts
            verts = (expand_batch(_r_mat, 5) @ verts) + _t_vec
            lines = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [2, 4], [3, 4]]
            colors = [[1, 0, 0]] * len(lines)

            cam = o3d.geometry.LineSet()
            cam.points = o3d.utility.Vector3dVector(verts.squeeze(-1))
            cam.lines = o3d.utility.Vector2iVector(lines)
            cam.colors = o3d.utility.Vector3dVector(colors)
            
            extrinsic = np.concatenate([_r_mat, _t_vec], axis=-1)
            extrinsic = np.concatenate([extrinsic, np.zeros([1, 4], np.float32)], axis=0)
            extrinsic[-1, -1] = 1.
            
            img_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(_img[..., ::-1].astype(np.uint8)),
                    o3d.geometry.Image(np.ones((_h, _w), np.float32)),
                    1.0, 2.0, False
                    )
            img_ = o3d.geometry.PointCloud.create_from_rgbd_image(
                    img_rgbd,
                    o3d.camera.PinholeCameraIntrinsic(_w, _h, _K[0, 0], _K[1, 1], _K[0, 2], _K[1, 2])
                    )
            #img_.transform(coord)
            img_.transform(extrinsic.astype(np.float64))
            return img_, cam

        num_points = self.inner_w * self.inner_h
        batch_K_inv = expand_batch(np.linalg.inv(mtx), num_points)

        for i, (img, rvec, tvec, imgpoint) in enumerate(zip(imgs, rvecs, tvecs, imgpoints)):
            r_mat = cv.Rodrigues(rvec)[0]
            Rt = np.concatenate([r_mat, tvec], -1)
            Rt = np.concatenate([Rt, np.zeros((1, 4))], 0)
            Rt[-1, -1] = 1.
            Rt = Rt @ coord
            r_mat = Rt[:3, :3]
            tvec = Rt[:3, -1]
            tvec = np.expand_dims(tvec, -1)

            imgpoint_ = np.concatenate([imgpoint, np.ones([imgpoint.shape[0], 1, 1], np.float32)], -1)
            
            campoint = batch_K_inv @ imgpoint_.transpose(0, 2, 1)
            worldpoint = expand_batch(r_mat.transpose(), num_points) @ campoint
            _Rt = r_mat.transpose() @ tvec
            
            s = worldpoint[:, -1] / _Rt[-1]
            s = np.expand_dims(np.repeat(s, 3, axis=1), -1)
            corners = (worldpoint / s - _Rt).squeeze(-1)

            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector(corners)
            line.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(num_points - 1)])
            LINE_COLORS = [[1, 0, 0], [1, 0.5, 0], [0.75, 0.75, 0], [0, 1, 0], [0, 0.75, 0.75], [0, 0, 1], [1, 0, 1]]
            
            line_colors = []
            for i in range(self.inner_w):
                for j in range(self.inner_h):
                    k = i % len(LINE_COLORS)
                    line_colors.append(LINE_COLORS[k])
                          
            line.colors = o3d.utility.Vector3dVector(line_colors[:-1])
            #line.transform(coord)
            vis.add_geometry(line)

            img_, cam = create_camera(img, r_mat, tvec, mtx)
            vis.add_geometry(cam)
            vis.add_geometry(img_)

        vis.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', 
                        help='input video for calibration')
    parser.add_argument('--output',
                        default='camera_parameters.npy',
                        help='npy file of camera parameters')
    parser.add_argument('--w',
                        type=int,
                        default=6,
                        help='the width of inner corners of chessboard')
    parser.add_argument('--h',
                        type=int,
                        default=8,
                        help='the height of inner corners of chessboard')
    parser.add_argument('--show',
                        action='store_true',
                        help='to show the 3D visualization of calibration (require install open3D and only work on Linux)')
    
    args = parser.parse_args()
    calibrator = Calibrator(args)
    calibrator.run()
