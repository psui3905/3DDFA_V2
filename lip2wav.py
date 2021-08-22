# coding: utf-8

__author__ = 'cleardusk'

import sys
import argparse
import cv2
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
#from utils.render_ctypes import render  # faster
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, get_suffix
from utils.tddfa_util import str2bool
import numpy as np
from numpy import savetxt
from sklearn import preprocessing
import pandas as pd
import os
import csv


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    for clip in os.listdir(args.path):
        for cut in os.listdir('/'.join([args.path, clip])):
            if len([name for name in os.listdir('/'.join([args.path, clip, cut])) if name.split('.')[-1] == 'csv']) == 1:
                for img_file in os.listdir('/'.join([args.path, clip, cut])):
                    if img_file.split('.')[-1] == 'jpg':
                        f_p = '/'.join([args.path, clip, cut, img_file])
                        
                        # Given a still image path and load to BGR channel
                        img = cv2.imread(f_p)
                        # Detect faces, get 3DMM params and roi boxes
                        boxes = face_boxes(img)
                        n = len(boxes)
                        if n == 0:
                            print('No face detected - {}'.format('/'.join([args.path, clip, cut, img_file.split('.')[0]]) + '.csv'))
                            continue
                        # print(f'Detect {n} faces')
                        param_lst, roi_box_lst = tddfa(img, boxes)
                        # Visualization and serialization
                        dense_flag = args.opt in ('2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')
                        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
                        ver_lst = np.array(ver_lst).flatten()
                        normal_lst = ver_lst/200
                        img_csv_path = '/'.join([args.path, clip, cut, img_file.split('.')[0]]) + '.csv'
                        if os.path.exists(img_csv_path):
                            os.remove(img_csv_path)
                        savetxt(img_csv_path, normal_lst, delimiter=',')
                        print('Saving {} to {}...'.format(img_file, img_csv_path))
                    # df = pd.DataFrame({"img" : img_file, "ver" : ver_lst}).to_csv(landmark_f_p)
                    # print('Origin Min: {} Max: {} - Min: {} Max: {}'.format(ver_lst.min(), ver_lst.max(), normal_lst.min(), normal_lst.max()))

def check(args):
    counter = 0
    total = 0
    counter_remain = False
    for clip in os.listdir(args.path):
        for cut in os.listdir('/'.join([args.path, clip])):
            if len([name for name in os.listdir('/'.join([args.path, clip, cut])) if name.split('.')[-1] == 'csv']) == 1:
                counter_remain = True
        if counter_remain:
            counter += 1
            counter_remain = False
            print(clip)
        total += 1
    print('Remains: {}/{}'.format(counter, total))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--img_fp', type=str, default='examples/inputs/trump_hillary.jpg')
    parser.add_argument('-m', '--mode', type=str, default='cpu', help='gpu or cpu mode')
    parser.add_argument('-p', '--path', type=str, default='/mnt/data/charles/Lip2Wav_baseline/Lip2Wav/Dataset/chem/preprocessed', help='path to chem dataset')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse',
                        choices=['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'pose', 'ply', 'obj'])
    parser.add_argument('--show_flag', type=str2bool, default='true', help='whether to show the visualization result')
    parser.add_argument('--onnx', action='store_true', default=True)
    args = parser.parse_args()
    main(args)
    # check(args)
