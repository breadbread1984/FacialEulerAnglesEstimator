#!/usr/bin/python3

from os.path import join;
from math import sqrt, atan2, sin, cos, asin;
import numpy as np;
import pickle;
import cv2;
import dlib;
from MTCNN import Detector;

class Landmarker(object):
    
    mapper = np.array([9, 18,19,20,21,22,23,24,25,26, \
                       27,28,29,30,31,32,33,34,35,36, \
                       37,38,39,40,41,42,43,44,45,46, \
                       47,48,49,50,51,52,53,54,55,56, \
                       57,58,59,60,62,63,64,66,67,68]);

    def __init__(self, model_path = 'models'):

        self.detector = Detector(model_path);
        self.landmarker = dlib.shape_predictor(join(model_path, 'shape_predictor_68_face_landmarks.dat'));
        with open(join(model_path, '68_world_pos.dat'), 'rb') as f:
            self.world_pos = np.array(pickle.loads(f.read()));

    def align(self, img):
        
        rectangles = self.detector.detect(img);
        landmarks = list();
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
        for rectangle in rectangles:
            upper_left = np.array(rectangle[0:2]);
            down_right = np.array(rectangle[2:4]);
            wh = down_right - upper_left;
            length = np.max(wh);
            center = (upper_left + down_right) / 2;
            upper_left = center - np.array([length, length]) / 2;
            down_right = upper_left + np.array([length, length]);
            rect = dlib.rectangle(int(upper_left[0]), int(upper_left[1]), int(down_right[0]), int(down_right[1]));
            sp = self.landmarker(rgb, rect);
            landmarks.append(([upper_left, down_right], [(p.x, p.y) for p in sp.parts()]));
        return landmarks;

    def euler2RotationMatrix(self, eulerAngles):

        s1 = sin(eulerAngles[0]);
        s2 = sin(eulerAngles[1]);
        s3 = sin(eulerAngles[2]);
        c1 = cos(eulerAngles[0]);
        c2 = cos(eulerAngles[1]);
        c3 = cos(eulerAngles[2]);
        rotation_matrix = np.array([
            [c2 * c3, -c2 * s3, s2],
            [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],
            [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2],
        ], dtype = np.float32);
        return rotation_matrix;

    def axisAngle2RotationMatrix(self, axis_angle):

        rotation_matrix = cv2.Rodrigues(axis_angle);
        return rotation_matrix;

    def rotationMatrix2Euler(self, rotation_matrix):

        q0 = sqrt(1 + rotation_matrix[0,0] + rotation_matrix[1,1] + rotation_matrix[2,2]) / 2.0;
        q1 = (rotation_matrix[2,1] - rotation_matrix[1,2]) / (4.0 * q0);
        q2 = (rotation_matrix[0,2] - rotation_matrix[2,0]) / (4.0 * q0);
        q3 = (rotation_matrix[1,0] - rotation_matrix[0,1]) / (4.0 * q0);

        t1 = 2.0 * (q0 * q2 + q1 * q3);
        if t1 > 1: t1 = 1.0;
        if t1 < -1: t1 = -1.0;
        yaw = asin(t1);
        pitch = atan2(2.0 * (q0 * q1 - q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3);
        roll = atan2(2.0 * (q0 * q3 - q1 * q2), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3);
        return (pitch, yaw, roll);

    def eulerAngles(self, landmarks, img_size):

        assert type(img_size) is tuple and len(img_size) == 2;
        # camera parameters
        cx = img_size[0] / 2.;
        cy = img_size[1] / 2.;
        fx = 500. * img_size[0] / 640.;
        fy = 500. * img_shape[1] / 480.;
        fx = (fx + fy) / 2.;
        fy = fx;
        # initial extrinsic estimate
        Z = fx / 1.0;
        X = (0. - cx) * (1.0 / fx) * Z;
        Y = (0. - cy) * (1.0 / fy) * Z;
        vec_trans = np.array([X,Y,Z]);
        vec_rot = np.array([0,0,0]);
        intrinsic = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype = np.float32);
        mapped_landmarks = np.array(landmarks)[self.mapper];
        rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints = self.world_pos, imagePoints = mapped_landmarks, cameraMatrix = intrinsic, distCoeffs = None, rvec = vec_rot, tvec = vec_trans, useExtrinsicGuess = True);
        z_x = sqrt(tvec[0] * tvec[0] + tvec[2] * tvec[2]);
        eul_x = atan2(tvec[1], z_x);
        z_y = sqrt(tvec[1] * tvec[1] + tvec[2] * tvec[2]);
        eul_y = -atan2(tvec[0], z_y);
        camera_ration = self.euler2RotationMatrix([eul_x, eul_y, 0]);
        head_rotation = self.axisAngle2RotationMatrix(rvec);
        corrected_rotation = camera_rotation * head_rotation;
        euler_corrected = self.rotationMatrix2Euler(corrected_rotation);
        return (tvec[0], rvec[1], tvec[2], euler_corrected[0], euler_corrected[1], euler_corrected[2]);

if __name__ == "__main__":

    import sys;
    if len(sys.argv) != 2:
        print('Usage: ' + sys.argv[0] + ' <image>');
        exit(1);
    img = cv2.imread(sys.argv[1]);
    if img is None:
        print('invalid image!');
        exit(1);
    landmarker = Landmarker();
    landmarks = landmarker.align(img);
    for landmark in landmarks:
        cv2.rectangle(img, tuple(landmark[0][0].astype('int32')), tuple(landmark[0][1].astype('int32')), (255,0,0), 2);
        for pts in landmark[1]:
            pts = (int(pts[0]),int(pts[1]));
            cv2.circle(img, pts, 2, (0, 255, 0), -1);
    cv2.imshow('landmarks', img);
    cv2.waitKey();

