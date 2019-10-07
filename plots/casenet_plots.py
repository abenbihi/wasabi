"""This plot script is taken from the official CaseNet code i.e. this is not
mine."""
#Copyright (c) 2017 Mitsubishi Electric Research Laboratories (MERL).   All rights reserved.
#
#The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications.  MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose.  In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.
#
#As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes. 

#visualize CASENet (Simple, fast)
# python visualize_multilabel.py ~/datasets/Cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png
#visualize CASENet (Full, slow)
# python visualize_multilabel.py ~/datasets/Cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png -g ~/datasets/Cityscapes/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_edge.bin
import os
import numpy as np
import numpy.random as npr
import cv2
import struct
import sys
import time
import multiprocessing
import argparse


def gen_hsv_class_cityscape():
    return np.array([
        359, # 0.road
        320, # 1.sidewalk
        40,  # 2.building
        80,  # 3.wall
        90,  # 4.fence
        10,  # 5.pole
        20,  # 6.traffic light
        30,  # 7.traffic sign
        140, # 8.vegetation
        340, # 9.terrain
        280, # 10.sky
        330, # 11.person
        350, # 12.rider
        120, # 13.car
        110, # 14.truck
        130, # 15.bus
        150, # 16.train
        160, # 17.motorcycle
        170  # 18.bike
    ])/2.0


def get_class_name_cityscape():
    return ['road',
            'sidewalk',
            'building',
            'wall',
            'fence',
            'pole',
            'traffic light',
            'traffic sign',
            'vegetation',
            'terrain',
            'sky',
            'person',
            'rider',
            'car',
            'truck',
            'bus',
            'train',
            'motorcycle',
            'bicycle']

def vis_multilabel(prob, img_h, img_w, K_class, hsv_class=None, use_white_background=False):
    label_hsv = np.zeros((img_h, img_w, 3), dtype=np.float32)
    prob_sum = np.zeros((img_h, img_w), dtype=np.float32)
    prob_edge = np.zeros((img_h, img_w), dtype=np.float32)

    use_abs_color = True

    i_color = 0
    for k in range(0, K_class):
        prob_k = prob[:, :, k].astype(np.float32)
        if prob_k.max() == 0:
            continue
        hi = hsv_class[ k if use_abs_color else i_color ]
        i_color += 1
        label_hsv[:, :, 0] += prob_k * hi  # H
        prob_sum += prob_k
        prob_edge = np.maximum(prob_edge, prob_k)

    prob_sum[prob_sum == 0] = 1.0
    label_hsv[:, :, 0] /= prob_sum
    if use_white_background:
        label_hsv[:, :, 1] = prob_edge * 255
        label_hsv[:, :, 2] = 255
    else:
        label_hsv[:, :, 1] = 255
        label_hsv[:, :, 2] = prob_edge * 255

    label_bgr = cv2.cvtColor(label_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return label_bgr
