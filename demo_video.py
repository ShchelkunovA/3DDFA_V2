# coding: utf-8

__author__ = 'cleardusk'

import argparse
import imageio
from tqdm import tqdm
import yaml

# from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
# from utils.render import render
from utils.functions import cv_draw_landmark, get_suffix
from utils import pose
import os
import importlib
import sys
from facenet_pytorch import MTCNN
import torch
import cv2


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        device = torch.device('cpu')
        face_boxes = MTCNN(keep_all=True, device=device, thresholds=[0.5, 0.6, 0.6])
        # model_path = 'E:\work\e-concierge\Pytorch_Retinaface'
        # sys.path.append(os.path.join(model_path, os.path.pardir))
        # det = importlib.import_module('Pytorch_Retinaface')
        # face_boxes = det.get_detector(r"E:\work\e-concierge\weights\mobilenet0.25_Final.pth", True, 0.8)
        # face_boxes = FaceBoxes()

    # Given a video path
    fn = args.video_fp.split('/')[-1]
    reader = imageio.get_reader(args.video_fp)

    fps = reader.get_meta_data()['fps']

    suffix = get_suffix(args.video_fp)
    video_wfp = f'examples/results/videos/{fn.replace(suffix, "")}_{args.opt}.mp4'
    writer = imageio.get_writer(video_wfp, fps=fps)

    # run
    dense_flag = args.opt in ('3d',)
    pre_ver = None
    for i, frame in tqdm(enumerate(reader)):
        frame_bgr = frame[..., ::-1]  # RGB->BGR

        if i == 0:
            # the first frame, detect face, here we only use the first face, you can change depending on your need
            boxes = face_boxes.detect(frame)
            if boxes[0] is None: continue
            boxes = boxes[0]
            param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # refine
            param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
        else:
            param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')

            roi_box = roi_box_lst[0]
            # todo: add confidence threshold to judge the tracking is failed
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                boxes = face_boxes.detect(frame_bgr)
                if boxes[0] is None: continue
                boxes = boxes[0]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

        pre_ver = ver  # for tracking

        if args.opt == '2d_sparse':
            res = cv_draw_landmark(frame_bgr, ver)
        elif args.opt == '3d':
            res = render(frame_bgr, [ver], tddfa.tri)
        else:
            raise ValueError(f'Unknown opt {args.opt}')
        # ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
        for param in param_lst:
            P, pose_out = pose.calc_pose(param)
            cv2.putText(res, f'yaw: {pose_out[0]:.1f}, pitch: {pose_out[1]:.1f}, roll: {pose_out[2]:.1f}', (100, res.shape[0] - 200), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0))
        # pose.viz_pose(res, param_lst, ver)
        cv2.imshow('res', res)
        cv2.waitKey(1)
        writer.append_data(res[..., ::-1])  # BGR->RGB

    writer.close()
    print(f'Dump to {video_wfp}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of video of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--video_fp', type=str)
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse', choices=['2d_sparse', '3d'])
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
