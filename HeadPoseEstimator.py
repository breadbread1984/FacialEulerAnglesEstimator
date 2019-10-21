#!/usr/bin/python3

import numpy as np;
import cv2;
import dlib;

class HeadPoseEstimator(object):

  cam_matrix = np.array([6.5308391993466671e+002, 0.0, 3.1950000000000000e+002, 0.0, 6.5308391993466671e+002, 2.3950000000000000e+002, 0.0, 0.0, 1.0]).reshape(3, 3).astype(np.float32);
  dist_coeffs = np.array([7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]).reshape(5, 1).astype(np.float32);
  object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]]);
  reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]]);
  line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]];

  def __init__(self, face_landmark_path = "models/shape_predictor_68_face_landmarks.dat"):

    self.predictor = dlib.shape_predictor(face_landmark_path);

  def shape_to_np(self, shape):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=np.int32);
    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
      coords[i] = (shape.part(i).x, shape.part(i).y);
    # return the list of (x, y)-coordinates
    return coords;

  def estimate(self, img, face_rect):

    shape = self.predictor(img, dlib.rectangle(int(face_rect[0]), int(face_rect[1]), int(face_rect[2]), int(face_rect[3])));
    shape = self.shape_to_np(shape);
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]]);
    _, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts, image_pts, self.cam_matrix, self.dist_coeffs);
    reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vec, translation_vec, self.cam_matrix,
                                        self.dist_coeffs);
    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)));
    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec);
    pose_mat = cv2.hconcat((rotation_mat, translation_vec));
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat);
    return shape, reprojectdst, euler_angle;

  def visualize(self, img, shape, reprojectdst, euler_angle):

    for (x, y) in shape:
      cv2.circle(img, (x, y), 1, (0, 0, 255), -1);
    for start, end in self.line_pairs:
      cv2.line(img, reprojectdst[start], reprojectdst[end], (0, 0, 255));
    cv2.putText(img, "pitch: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), thickness=2);
    cv2.putText(img, "yaw: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), thickness=2);
    cv2.putText(img, "roll: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), thickness=2);
    return img;

if __name__ == "__main__":

  import sys;
  from MTCNN import Detector;
  if len(sys.argv) != 2:
    print('Usage: ' + sys.argv[0] + " <video>");
    exit(1);
  detector = Detector();
  estimator = HeadPoseEstimator();
  cap = cv2.VideoCapture(sys.argv[1]);
  if cap is None:
    print('invalid video!');
    exit(1);
  while True:
    ret, img = cap.read();
    if ret == False: break;
    rectangles = detector.detect(img);
    estimation = [estimator.estimate(img,rectangle) for rectangle in rectangles];
    for shape, reprojectdst, euler_angle in estimation:
      img = estimator.visualize(img, shape, reprojectdst, euler_angle);
    cv2.imshow('headpose', img);
    cv2.waitKey(25);

