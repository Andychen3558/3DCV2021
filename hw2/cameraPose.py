from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import cv2
import time


def average(x):
    return list(np.mean(x,axis=0))

def cosine(v0, v1):
    return np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def trilaterate3D(x, lengths):
    p1 = x[0]
    p2 = x[1]
    p3 = x[2]
    r1 = lengths[0]
    r2 = lengths[1]
    r3 = lengths[2]
    e_x = (p2-p1) / np.linalg.norm(p2-p1)
    i = np.dot(e_x,(p3-p1))
    e_y = (p3-p1-(i*e_x)) / (np.linalg.norm(p3-p1-(i*e_x)))
    e_z = np.cross(e_x,e_y)
    d = np.linalg.norm(p2-p1)
    j = np.dot(e_y,(p3-p1))
    x = ((r1**2)-(r2**2)+(d**2)) / (2*d)
    y = (((r1**2)-(r3**2)+(i**2) + (j**2))/(2*j))-((i/j)*(x))
    z1 = np.sqrt(r1**2-x**2-y**2)
    z2 = np.sqrt(r1**2-x**2-y**2) * (-1)
    ans1 = p1+(x*e_x)+(y*e_y)+(z1*e_z)
    ans2 = p1+(x*e_x)+(y*e_y)+(z2*e_z)
    return [ans1, ans2]

def R_2vect(vector_orig, vector_fin):
    ## Convert the vectors to unit vectors.
    vector_orig = vector_orig / np.linalg.norm(vector_orig)
    vector_fin = vector_fin / np.linalg.norm(vector_fin)

    ## The rotation axis (normalised).
    axis = np.cross(vector_orig, vector_fin)
    axis_len = np.linalg.norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len

    ## Alias the axis coordinates.
    x = axis[0]
    y = axis[1]
    z = axis[2]

    ## The rotation angle.
    ca = cosine(vector_orig, vector_fin)
    sa = np.sqrt(1 - ca**2)

    ## Calculate the rotation matrix elements.
    R = np.zeros(shape=(3, 3))
    R[0,0] = 1.0 + (1.0 - ca)*(x**2 - 1.0)
    R[0,1] = -z*sa + (1.0 - ca)*x*y
    R[0,2] = y*sa + (1.0 - ca)*x*z
    R[1,0] = z*sa+(1.0 - ca)*x*y
    R[1,1] = 1.0 + (1.0 - ca)*(y**2 - 1.0)
    R[1,2] = -x*sa+(1.0 - ca)*y*z
    R[2,0] = -y*sa+(1.0 - ca)*x*z
    R[2,1] = x*sa+(1.0 - ca)*y*z
    R[2,2] = 1.0 + (1.0 - ca)*(z**2 - 1.0)

    return R

def solve_for_lengths(cos, dist):
    K1, K2 = (dist[2]/dist[1])**2, (dist[2]/dist[0])**2

    coeff4 = (K1*K2-K1-K2)**2 - 4*K1*K2*(cos[2]**2)
    coeff3 = 4 * (K1*K2-K1-K2) * K2 * (1-K1) * cos[0] \
            + 4*K1*cos[2] * ((K1*K2-K1+K2)*cos[1] + 2*K2*cos[0]*cos[2])
    coeff2 = (2*K2*(1-K1)*cos[0])**2 \
            + 2 * (K1*K2-K1-K2) * (K1*K2+K1-K2) \
            + 4*K1 * ((K1-K2) * (cos[2]**2) + K1*(1-K2)*(cos[1]**2) - 2*(1+K1)*K2*cos[0]*cos[1]*cos[2])
    coeff1 = 4 * (K1*K2+K1-K2) * K2 * (1-K1) * cos[0] \
            + 4*K1 * ((K1*K2-K1+K2) * cos[1]*cos[2] + 2*K1*K2*cos[0]*(cos[1]**2))
    coeff0 = (K1*K2+K1-K2)**2 - 4*(K1**2)*K2*(cos[1]**2)
    
    if coeff4 == 0:
        return
    # print(coeff4, coeff3, coeff2, coeff1, coeff0)

    x_roots = np.roots([coeff4, coeff3, coeff2, coeff1, coeff0])
    
    ## compute lengths for all roots
    lengths = []
    for root in x_roots:
        if np.isreal(root) == False:
            continue
        root = np.real(root)
        ### compute a, b
        a = np.sqrt((dist[0]**2) / (1 + (root)**2 - 2*root*cos[0]))
        b = root * a
        ### compute c
        m, m_prime = 1-K1, 1
        p, p_prime = 2 * (K1*cos[1] - root*cos[2]), 2 * (-root*cos[2])
        q, q_prime = root**2 - K1, (root**2)*(1-K2) + 2*root*K2*cos[0] - K2
        y = (m*q_prime - m_prime*q) / (p*m_prime - p_prime*m)
        c = y * a
        lengths.append([a, b, c])

    return lengths


def solveP3P(points3D, points2D, cameraMatrix, distCoeffs, numValid):
    ## Sample correspondences
    numValid = 100
    indices = np.random.choice(len(points3D), 3+numValid)
    trainIdx, validIdx = indices[:3], indices[3:]
    # trainIdx, validIdx = [0, 1, 2], [3, 4, 5, 6, 7]
    x, u = points3D[trainIdx], points2D[trainIdx]

    ## Verify that world points are not colinear
    if np.linalg.norm(np.cross(x[1]-x[0], x[2]-x[0])) < 1e-5:
        print("Point A, B, C are colinear!")
        return False, []

    ## Compute angle cosines and distances then obtain lengths
    v = np.dot(np.linalg.inv(cameraMatrix), np.hstack((u, np.ones((u.shape[0], 1)))).T).T
    cosines = cosine(v[0], v[1]), cosine(v[0], v[2]), cosine(v[1], v[2])
    distances = np.linalg.norm(x[0] - x[1]), np.linalg.norm(x[0] - x[2]), np.linalg.norm(x[1] - x[2])
    lengths = solve_for_lengths(cosines, distances)
    if not lengths:
        return False, []
    
    ## Solve 3D-3D problem
    Rot, T = [], []
    Lambda = []
    for i in range(len(lengths)):
        ans = trilaterate3D(x, lengths[i])
        for t in ans:
            l = np.linalg.norm(x[0] - t) / np.linalg.norm(v[0])
            if l < 0:
                continue

            r = R_2vect(x[0] - t, l*v[0])
            if np.linalg.det(r) - 1 > 1e-5:
                continue

            Rot.append(r)
            T.append(t)
            Lambda.append(l)

    ## Utilize other correspondences to select the best solution
    best_R, best_T, best_error = 0, 0, float('inf')
    x_val, u_val = points3D[validIdx], points2D[validIdx]
    v_val = np.dot(np.linalg.inv(cameraMatrix), np.hstack((u_val, np.ones((u_val.shape[0], 1)))).T).T
    for i in range(len(Rot)):
        err = 0
        for j in range(len(x_val)):
            err += np.linalg.norm(Rot[i].dot(x_val[j]-T[i]) - Lambda[i] * v_val[j])
        error = err/len(x_val)
        if error < best_error:
            best_error = error
            best_R = Rot[i]
            best_T = T[i]
    # print("Best error = {}".format(best_error))
    # print("Best R = {}".format(best_R))
    # print("Best T = {}".format(best_T))
    return True, [best_error, best_R, best_T]


def p3psolver(query, model, cameraMatrix, distortion, numIter=100, numValid=100):
    kp_query, desc_query = query
    kp_model, desc_model = model

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query, desc_model, k=2)

    gmatches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            gmatches.append(m)

    points2D = np.empty((0,2))
    points3D = np.empty((0,3))

    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D, kp_query[query_idx]))
        points3D = np.vstack((points3D, kp_model[model_idx]))

    # return cv2.solvePnPRansac(points3D, points2D, cameraMatrix, distCoeffs)

    ## Apply RANSAN algorithm
    best_R, best_T, best_error = 0, 0, float('inf')
    for i in range(numIter):
        isValid, rst = solveP3P(points3D, points2D, cameraMatrix, distCoeffs, numValid)
        if isValid:
            err, R, T = rst
            if err < best_error:
                best_error = err
                best_R = R
                best_T = T
    # print("------")
    # print("Final best error = {}".format(best_error))
    # print("Best R = {}".format(best_R))
    # print("Best T = {}".format(best_T))
    return best_R, best_T


if __name__ == '__main__':

    ## Read data
    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")

    ## Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)


    validIdxHead = 164
    errorR, errorT = [], []
    Rotation, Translation = [], []
    for idx in range(validIdxHead, len(images_df)+1):
        ## Load query image
        fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
        rimg = cv2.imread("data/frames/"+fname,cv2.IMREAD_GRAYSCALE)

        ## Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"]==idx]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        ## Find correspondance and solve pnp
        cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])    
        distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])
        
        rmat, tvec = p3psolver((kp_query, desc_query), (kp_model, desc_model), cameraMatrix, distCoeffs)

        ## Get camera pose groudtruth 
        ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
        rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
        tvec_gt = ground_truth[["TX","TY","TZ"]].values
        # print(rotq_gt, tvec_gt)

        ## Compute the median pose error
        ### Rotation part
        rotq = R.from_matrix(rmat).as_quat()
        Rotation.append(rotq)

        rotq = R.from_quat(rotq)
        angle = R.from_quat(rotq_gt) * rotq.inv()
        errorR.append(np.linalg.norm(angle.as_rotvec()))
        ### Translation part
        Translation.append(tvec)
        errorT.append(np.linalg.norm(tvec - tvec_gt))

    ## save rotation and translation for all queries
    np.save('./Rotation.npy', np.array(Rotation))
    np.save('./Translation.npy', np.array(Translation))

    print("Median pose error is [R = {}, T = {}]".format(errorR[len(errorR)//2], errorT[len(errorT)//2]))
