"""Image retrieval with semantic edge descriptor. 
Runs the retrieval and computes the metrics.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import multiprocessing as mp
import os
import pickle
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

import datasets.survey
import datasets.retrieval
import methods.wasabi.retrieve as wasabi_retrieve

def describe(args):
    """Runs retrieval.

    Args:
        args: retrieval parameters
        retrieval: retrieval instance
        n_value: values at which you compute recalls.
    """
    global_start_time = time.time()
    res_dir = "res/wasabi/%d/retrieval/"%args.trial
    
    # load db traversal
    surveyFactory = datasets.survey.SurveyFactory()
    meta_fn = "%s/%d/c%d_db.txt"%(args.meta_dir, args.slice_id, args.cam_id)
    kwargs = {"meta_fn": meta_fn, "img_dir": args.img_dir, "seg_dir": args.seg_dir}
    db_survey = surveyFactory.create(args.data, **kwargs)
    
    # describe db img
    print('\n** Compute des for database img **')
    local_start_time = time.time()
    db_des_fn = '%s/%d_c%d_db.pickle'%(res_dir, args.slice_id, args.cam_id)

    img_des_l = []
    if os.path.exists(db_des_fn): # if you did not compute the db already
        os.remove(db_des_fn)
    db_size = db_survey.get_size()
    for idx in range(db_size):
        print("%d/%d"%(idx, db_size))
        _, img_des = wasabi_retrieve.describe_img_from_survey(args, db_survey, idx)
        img_des_l.append(img_des)
    with open(db_des_fn, 'wb') as f:
        pickle.dump(img_des_l, f)
    duration = (time.time() - local_start_time)
    print('(END) run time: %d:%02d'%(duration/60, duration%60))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=int, required=True)
    parser.add_argument('--top_k', type=int, default=20)

    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--seg_dir', type=str, required=True)
    parser.add_argument('--meta_dir', type=str, required=True)

    parser.add_argument('--dist_pos', type=float, required=True)
    parser.add_argument('--min_blob_size', type=int, default=50)
    parser.add_argument('--min_contour_size', type=int, default=50)
    parser.add_argument('--min_edge_size', type=int, default=40)
    parser.add_argument('--contour_sample_num', type=int, default=64)
    
    parser.add_argument('--slice_id', type=int, default=22)
    parser.add_argument('--cam_id', type=int, default=0)
    parser.add_argument('--survey_id', type=int, default=0)
    parser.add_argument('--data', type=str, required=True, help='{cmu, lake}')
    args = parser.parse_args()
    
    res_dir = "res/wasabi/%d/retrieval/"%args.trial
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    describe(args)


