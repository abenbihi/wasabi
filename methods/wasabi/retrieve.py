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
from tools import cst, edge_descriptor, semantic_proc, contour_proc


#class DesParams(object):
#    def __init__(self):
#        self.min_blob_size = 50
#        self.min_contour_size = 50
#        self.contour_sample_num = 64
#        self.top_k = 20
#        self.trial = 1
#        self.min_edge_size = 50

def extract_semantic_edge(params, sem_img):
    """Trial1 on semantic edge extraction."""
    debug = (0==1)
    if debug:
        colors_v = np.loadtxt('meta/color_mat.txt').astype(np.uint8)
    
    # convert semantic img to label img
    lab_img = semantic_proc.col2lab(sem_img)
    lab_img_new = semantic_proc.merge_small_blobs(lab_img)
    #if debug:
    #    sem_img_new = semantic_proc.lab2col(lab_img_new)
    #    cv2.imshow('sem_img_new', sem_img_new)

    # get semantic contours
    label_l = np.unique(lab_img_new)
    contours_d = {}
    for label in label_l:
        l,c = np.where(lab_img_new==label)
        mask = np.zeros(lab_img_new.shape).astype(np.uint8)
        mask[l,c] = 255
        im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, 
                cv2.CHAIN_APPROX_NONE)
        # keep only long contours 
        big_contours_l = []
        for c in contours:
            if c.shape[0] > params.min_contour_size:
                big_contours_l.append(c)
        if len(big_contours_l)>0:
            contours_d[label] = np.vstack(big_contours_l)
    
    ## draw semantic contours
    #if debug:
    #    debug_img = np.zeros(sem_img.shape, np.uint8)
    #    for label, contour in contours_d.items():
    #        print('label: %d\t%s'%(label, label_name[label]))
    #        contour = np.squeeze(contour)
    #        debug_img[contour[:,1], contour[:,0]] = colors_v[label]
    #        cv2.imshow('debug_img', debug_img)
    #        if (cv2.waitKey(0) & 0xFF ) == ord("q"):
    #            exit(0)

    # extract connected components from the contour
    curves_d = {}
    for label, contour in contours_d.items():
        curves = contour_proc.contours2curve(params, contour, sem_img)
        if curves is not None:
            curves_d[label] = curves
    
    ## draw semantic curves
    #if debug:
    #    for label, curves_l in curves_d.items():
    #        print('Edge label: %d\t%s'%(label, label_name[label]))
    #        curve_img = np.zeros(sem_img.shape, np.uint8)
    #        for i, curve in enumerate(curves_l):
    #            curve_img[curve[:,1], curve[:,0]] = colors_v[label]
    #            cv2.imshow('curve_img', curve_img)
    #            cv2.waitKey(0)
    #    if (cv2.waitKey(0) & 0xFF) == ord("q"):
    #        exit(0)

    return curves_d


def describe_semantic_edge(params, curves_d):
    des_d = {}
    for label_id, curves_l in curves_d.items():
        des_d[label_id] = []
        for c in curves_l:
            des_d[label_id].append(
                    edge_descriptor.wavelet_chuang96(params, c, None))
    return des_d


#def describe_img(args, sem_img):
#    """(Deprecated) Computes global image descriptor.
#
#    Args:
#        sem_img: semantic img
#
#    Returns:
#        des_d: dict where keys are labels, values are edge descriptors.
#    """
#    sem_curves_d = extract_semantic_edge(args, sem_img)
#    des_d = describe_semantic_edge(args, sem_curves_d)
#    return des_d


#def draw_semantic_curves(curves_d, palette, img_shape):
#    curve_img = 255*np.ones(img_shape, np.uint8)
#    #curve_img = np.zeros(img_shape, np.uint8)
#    for label, curves_l in curves_d.items():
#        for curve in curves_l:
#            curve_img[curve[:,1], curve[:,0]] = palette[label]
#    return curve_img


#def get_db_des(args, db_survey, n_values):
#    """(Deprecated) Computes/load database image descriptors.
#
#    Args:
#        args: retrieval parameters
#        retrieval: retrieval instance
#        n_value: values at which you compute recalls.
#    """
#    global_start_time = time.time()
#    res_dir = "res/wasabi/%d/retrieval/"%args.trial
#    db_fn = '%s/%d_c%d_db.pickle'%(res_dir, args.slice_id, args.cam_id)
#    
#    if not os.path.exists(db_fn): # if you did not compute the db already
#        print('\n** Compute des for database img **')
#        des_l = [] # list of dict. Each dist describes an img.
#        db_size = db_survey.get_size()
#        for i in range(db_size):
#            if i%20==0:
#                duration = (time.time() - global_start_time)
#                print('%d/%d\tglobal run time: %d:%02d'%(i, db_size, duration/60, duration%60))
#            
#            sem_img = db_survey.get_semantic_img(i)
#            if sem_img is None:
#                raise ValueError("There is no db image with index %d"%idx)
#            des_l.append(describe_img(args, sem_img))
#
#        with open(db_fn, 'wb') as f:
#            pickle.dump(des_l, f)
#    else: # if you already computed it, load it from disk
#        print('\n** Load des for database img **')
#        with open(db_fn, 'rb') as f:
#            des_l = pickle.load(f)
#    
#    duration = (time.time() - global_start_time)
#    print('END: global run time: %d:%02d'%(duration/60, duration%60))
#    return des_l


def describe_img_from_survey(args, survey, idx):
    """Computes global image descriptor.

    Args:
        sem_img: semantic img

    Returns:
        des_d: dict where keys are labels, values are edge descriptors.
    """
    sem_img = survey.get_semantic_img(idx)
    sem_curves_d = extract_semantic_edge(args, sem_img)
    des_d = describe_semantic_edge(args, sem_curves_d)
    return idx, des_d


def collect_restults(result):
    global result_l
    result_l.append(result)


def get_img_des_parallel(args, survey):
    """Computes/load database image descriptors (parallel processing).

    Example taken from
    https://www.machinelearningplus.com/python/parallel-processing-python/

    Args:
        args: retrieval parameters
        retrieval: retrieval instance
        n_value: values at which you compute recalls.
    """
    pool = mp.Pool(mp.cpu_count())
    result_l = []
    for idx in range(survey.get_size()):
        pool.apply_async(describe_img_from_survey, args=(args, survey, idx))
    pool.close()
    pool.join()

    result_l.sort(key=lambda x: x[0])
    des_l = [r for i, r in result_l]
    return des_l


def retrieve(args, q_img_des, db_img_des_l):
    """ """
    q_label_v = np.array(list(q_img_des.keys()))
    db_size = len(db_img_des_l)
    img_distance_v = 1e4*np.ones(db_size)

    # compute distances with db imgs
    for db_idx in range(db_size):
        db_img_des = db_img_des_l[db_idx] # dict of edge descriptor

        # check if db and q images have labels in common
        db_label_v = np.array(list(db_img_des.keys()))
        matching_label_v = q_label_v[np.in1d(q_label_v, db_label_v)]
        if matching_label_v.size != 0: # yes, they do
            img_distance_v[db_idx] = 0. # init img distance
        else: # no, so let the image distance to 1e4
            continue

        # image distance
        curve_count = 0
        for label in matching_label_v:
            # get the list of edge descriptor in each class in common
            q_edge_des_l = q_img_des[label]
            db_edge_des_l = db_img_des[label]
            
            # compute the distance between all pairs of edges of the same label
            q_des_v, db_des_v = np.array(q_des_l), np.array(db_des_l)
            q_des_v = np.expand_dims(q_des_v, 1)
            db_des_v = np.expand_dims(db_des_v, 0)
            d_des = np.linalg.norm(q_des_v - db_des_v, ord=None, axis=2)
            
            # associate edge pairs to minimimse the pair distance
            row_ind, col_ind = linear_sum_assignment(d_des)
            img_distance_v[db_idx] += np.sum(d_des[row_ind, col_ind])
            curve_count += row_ind.shape[0]
        
        # average the edge descriptor distances
        if curve_count != 0:
            img_distance_v[db_idx] = img_distance_v[db_idx]/float(curve_count)
           
    # db image idx from nearest to furthest in descriptor space
    order = np.argsort(img_distance_v)
    return idx, order


def retrieve_parallel(args, q_img_des_l, db_img_des_l):
    """ """
    q_size = len(q_img_des_l)
    pool = mp.Pool(mp.cpu_count())
    result_l = []
    for q_idx in range(q_size):
        pool.apply_async(retrieve, args=(args, q_idx, q_img_des_l[q_idx],
            db_img_des_l))
    pool.close()
    pool.join()

    result_l.sort(key=lambda x: x[0])
    order_l = [r for i, r in result_l]
    return order_l


#def retrieve_img(args, q_survey, idx, db_des_l):
#    sem_img = q_survey.get_semantic_img(idx)
#    sem_curves_d = extract_semantic_edge(args, sem_img) # get semantic contours and describe them
#    q_des_d = describe_semantic_edge(args, sem_curves_d)
#    q_label_v = np.array(list(q_des_d.keys()))
#
#    # compute distances with db imgs
#    img_distance_v = 1e4*np.ones(db_size)
#    for db_idx in range(db_size):
#        db_label_v = np.array(list(db_sem_curves_d.keys()))
#        matching_label_v = q_label_v[np.in1d(q_label_v, db_label_v)]
#        if matching_label_v.size != 0:
#            img_distance_v[db_idx] = 0. # init img distance
#        else: # no common label between q and db
#            continue
#        # image distance
#        curve_num = 0
#        for label in matching_label_v:
#            q_des_l, db_des_l = q_des_d[label], db_des_d[label]
#            q_des_v, db_des_v = np.array(q_des_l), np.array(db_des_l)
#            q_des_v = np.expand_dims(q_des_v, 1)
#            db_des_v = np.expand_dims(db_des_v, 0)
#            d_des = np.linalg.norm(q_des_v - db_des_v, ord=None, axis=2)
#            
#            row_ind, col_ind = linear_sum_assignment(d_des)
#            img_distance_v[db_idx] += np.sum(d_des[row_ind, col_ind])# /row_ind.shape[0]
#            #img_distance_v[db_idx] += np.sum(d_des[row_ind, col_ind])
#            curve_num += row_ind.shape[0]
#    
#        if curve_num != 0:
#            img_distance_v[db_idx] = img_distance_v[db_idx]/float(curve_num)
#            
#    order = np.argsort(img_distance_v) # idx of db img from n
#    return order


def bench(args, n_values):
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
    
    # load query traversal
    meta_fn = "%s/%d/c%d_%d.txt"%(args.meta_dir, args.slice_id, args.cam_id, 
            args.survey_id)
    kwargs["meta_fn"] = meta_fn
    q_survey = surveyFactory.create(args.data, **kwargs)
    
    # describe db img
    print('\n** Compute des for database img **')
    db_des_fn = '%s/%d_c%d_db.pickle'%(res_dir, args.slice_id, args.cam_id)
    if not os.path.exists(db_des_fn): # if you did not compute the db already
        db_img_des_l = get_img_des_parallel(args, db_survey)
        with open(db_des_fn, 'wb') as f:
            pickle.dump(db_img_des_l, f)
    else: # if you already computed it, load it from disk
        print('\n** Load des for database img **')
        with open(db_des_fn, 'rb') as f:
            db_img_des_l = pickle.load(f)
    duration = (time.time() - global_start_time)
    print('END: database descriptors: %d:%02d'%(duration/60, duration%60))


    # describe q img
    print('\n** Compute des for query img **')
    q_img_des_l = get_img_des_parallel(args, q_survey)
    duration = (time.time() - global_start_time)
    print('END: query descriptors: %d:%02d'%(duration/60, duration%60))


    # retrieve each query
    order_l = retrieve_parallel(args, q_img_des_l, db_img_des_l)
    

    ## compute perf
    #retrieval = datasets.retrieval.Retrieval(db_survey, q_survey, dist_pos)
    #rank_l = retrieval.get_retrieval_rank(order_l, args.top_k)
    #gt_name_d = retrieval.get_gt_rank("name")
    #mAP = metrics.mAP(rank_l, gt_name_d)

    #gt_idx_l = retrieval.get_gt_rank("idx")
    #recalls = metrics.recallN(order_l, gt_idx_l, n_values)

    duration = (time.time() - global_start_time)
    print('END: Retrieval\tglobal run time: %d:%02d'%(duration/60, duration%60))

    return mAP, recalls


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

    n_values = [1, 5, 10, 20]
    bench(args, n_values)
    

    ##sem_curve_asso(slice_id, cam_id, survey_id)

    ##fuck1(args, slice_id, cam_id, survey_id)
    ##
    #if args.mode == 'fuse':
    #    #fuse_part_result(args)
    #    fuse_result_lake(args)
    #elif args.mode == 'fuck1':
    #    fuck1(args, args.slice_id, args.cam_id, args.survey_id)
    #else:
    #    print("Error: you came to the wrong mode motherfucker.")
    #    exit(1)
    

