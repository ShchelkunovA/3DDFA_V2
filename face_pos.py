# coding: utf-8
import yaml
# from FaceBoxes import FaceBoxes
from .TDDFA import TDDFA
from .utils.functions import cv_draw_landmark, get_suffix
import os
import sys
import torch
import cv2
import numpy as np
from math import cos, sin, atan2, asin, sqrt

def P2sRt(P):
    """ decompositing camera matrix P.
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation.
    """
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d


def matrix2angle(R):
    """ compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    refined by: https://stackoverflow.com/questions/43364900/rotation-matrix-to-euler-angles-with-opencv
    todo: check and debug
     Args:
         R: (3,3). rotation matrix
     Returns:
         x: yaw
         y: pitch
         z: roll
     """
    if R[2, 0] > 0.998:
        z = 0
        x = np.pi / 2
        y = z + atan2(-R[0, 1], -R[0, 2])
    elif R[2, 0] < -0.998:
        z = 0
        x = -np.pi / 2
        y = -z + atan2(R[0, 1], R[0, 2])
    else:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    return x, y, z


def calc_pose(param):
    P = param[:12].reshape(3, -1)  # camera matrix
    s, R, t3d = P2sRt(P)
    P = np.concatenate((R, t3d.reshape(3, -1)), axis=1)  # without scale
    pose = matrix2angle(R)
    pose = [p * 180 / np.pi for p in pose]

    return P, pose

class Pose_estimator():
    def __init__(self, config='configs/mb1_120x120.yml', mode='cpu', opt='2d_sparse', onnx=False):
        self.config = config
        self.mode = mode
        self.opt = opt
        self.onnx = onnx
        self.cfg = yaml.load(open(config), Loader=yaml.SafeLoader)
        if onnx:
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['OMP_NUM_THREADS'] = '4'

            from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
            from TDDFA_ONNX import TDDFA_ONNX

            # face_boxes = FaceBoxes_ONNX()
            self.tddfa = TDDFA_ONNX(**self.cfg)
        else:
            gpu_mode = mode == 'gpu'
            self.tddfa = TDDFA(gpu_mode=gpu_mode, **self.cfg)
            self.device = torch.device(mode)
        self.i = 0
        self.pre_ver = []
    def get_pose_angless(self, frame, boxes):
        frame_bgr = frame
        dense_flag = self.opt in ('3d',)
        angless = []

        param_lst, roi_box_lst = self.tddfa(frame_bgr, boxes)
        ver = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

        # refine
        param_lst, roi_box_lst = self.tddfa(frame_bgr, [ver], crop_policy='landmark')
        # ver = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
        for param in param_lst:
            P, pose_out = calc_pose(param)
            angless.append(pose_out)
        return angless