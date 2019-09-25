"""Test various step of image retrieval with semantic edge descriptor."""
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
from tools import cst, edge_descriptor, semantic_proc, contour_proc, metrics
import methods.wasabi.retrieve as wasabi_retrieve


def test(args):
    global_start_time = time.time()
    res_dir = "res/wasabi/%d/retrieval/"%args.trial
    colors_v = np.loadtxt('meta/color_mat.txt').astype(np.uint8)
    patch_shape = (200, 200, 3)
    
    # load db traversal
    surveyFactory = datasets.survey.SurveyFactory()
    meta_fn = "%s/%d/c%d_db.txt"%(args.meta_dir, args.slice_id, args.cam_id)
    kwargs = {"meta_fn": meta_fn, "img_dir": args.img_dir, "seg_dir": args.seg_dir}
    db_survey = surveyFactory.create(args.data, **kwargs)
    
    # load query traversal
    meta_fn = "%s/%d/c%d_%d.txt"%(args.meta_dir, args.slice_id, args.cam_id, 
            args.survey_id)
    kwargs["meta_fn"] = meta_fn
    q_survey = surveyFactory.create(args.data, **kwargs)

    retrieval = datasets.retrieval.Retrieval(db_survey, q_survey, args.dist_pos)
    q_idx_v, db_idx_v, label_v = retrieval._gen_game(args.dist_pos)
    
    
    for q_idx in q_idx_v:
        db_idx = db_idx_v[q_idx]
        label = label_v[q_idx]
        print('\n\nq_idx: %d\tdb_idx: %d\tlabel: %d'%(q_idx, db_idx, label))
        
        # describe q img
        q_img = q_survey.get_img(q_idx)
        q_sem_img = q_survey.get_semantic_img(q_idx)
        q_sem_edge_d = wasabi_retrieve.extract_semantic_edge(args, q_sem_img)
        q_img_des_d = wasabi_retrieve.describe_semantic_edge(args, q_sem_edge_d)
        q_curve_img = wasabi_retrieve.draw_semantic_curves(q_sem_edge_d,
                colors_v, q_sem_img.shape)       
        
        
        # describe db img
        db_img = db_survey.get_img(db_idx)
        db_sem_img = db_survey.get_semantic_img(db_idx)
        db_sem_edge_d = wasabi_retrieve.extract_semantic_edge(args, db_sem_img)
        db_img_des_d = wasabi_retrieve.describe_semantic_edge(args, db_sem_edge_d)
        db_curve_img = wasabi_retrieve.draw_semantic_curves(db_sem_edge_d,
                colors_v, db_sem_img.shape)       

        
        # display img, sem. img, sem. edge
        out_sem_img = np.hstack((q_sem_img, db_sem_img))
        out_curve_img = np.hstack((q_curve_img, db_curve_img))
        cv2.imshow('img q | db', np.hstack((q_img, db_img)))
        cv2.imshow('sem q | db', np.vstack((out_sem_img, out_curve_img)) )
        if (cv2.waitKey(0) & 0xFF) == ord("q"):
           exit(0)

        
        # get labels common to both img
        q_label_v = np.array(list(q_img_des_d.keys()))
        db_label_v = np.array(list(db_img_des_d.keys()))
        matching_label_v = q_label_v[np.in1d(q_label_v, db_label_v)]
        if matching_label_v.size == 0: # yes, they do
            continue
       
        
        # display edge for each label
        for label in matching_label_v:
            q_edge_l = q_sem_edge_d[label]
            db_edge_l = db_sem_edge_d[label]

            # (debug) draw the semantic curves to match
            q_edge_norm_l, q_patch_l = contour_proc.gen_patches(q_edge_l, 
                    (200,200,3), colors_v[label])
            db_edge_norm_l, db_patch_l = contour_proc.gen_patches(db_edge_l, 
                    (200,200,3), colors_v[label])
            q_patches = contour_proc.fuse_patches(q_patch_l)
            db_patches = contour_proc.fuse_patches(db_patch_l)
            cv2.imshow('patches q', q_patches)
            cv2.imshow('db_patches', db_patches)
            if (cv2.waitKey(0) & 0xFF) == ord("q"):
                exit(0)

            # describe and match of this label
            q_des_v = np.expand_dims(np.array(q_img_des_d[label]), 1)
            db_des_v = np.expand_dims(np.array(db_img_des_d[label]), 0)
            dist = np.linalg.norm(q_des_v - db_des_v, ord=None, axis=2)
            order = np.argsort(dist, axis=1)
            match = order[:,0]
            
            # show curve association
            row_ind, col_ind = linear_sum_assignment(dist)
            assignment = np.zeros((len(q_edge_l), len(db_edge_l)), np.uint8)
            assignment[row_ind, col_ind] = 1
            print(dist)
            print(assignment)

            # show similarity order
            for i, q_patch in enumerate(q_patch_l):
                ordered_db_patch_l = [db_patch_l[j] for j in order[i,:]]
                d_l = ['%.3f'%dist[i,j] for j in order[i,:] ]
                out = np.hstack((q_patch, np.hstack(ordered_db_patch_l)))
                for j in range(1, len(ordered_db_patch_l)+1):
                    out[:,j*patch_shape[1]] = 255
                print('d: ' + ' '.join(d_l))
                cv2.imshow('match q | db', out)
                if (cv2.waitKey(0) & 0xFF) == ord("q"):
                    exit(0)


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
    
    test(args)
