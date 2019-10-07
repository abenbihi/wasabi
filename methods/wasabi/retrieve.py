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
from tools import cst, edge_descriptor, semantic_proc, edge_proc, metrics

# global variable
result_l = []

def describe_img_from_survey(args, survey, idx):
    """Computes global image descriptor.

    Args:
        survey: traversal.
        idx: index of the survey image to describe.

    Returns:
        des_d: dict where keys are labels, values are edge descriptors.
    """
    sem_img = survey.get_semantic_img(idx)
    sem_edge_d = semantic_proc.extract_semantic_edge(args, sem_img)
    des_d = edge_descriptor.describe_semantic_edge(args, sem_edge_d)
    return idx, des_d


def describe_img(args, sem_img):
    """Computes global image descriptor.

    Args:
        sem_img: semantic img

    Returns:
        des_d: dict where keys are labels, values are edge descriptors.
    """
    sem_edge_d = semantic_proc.extract_semantic_edge(args, sem_img)
    des_d = edge_descriptor.describe_semantic_edge(args, sem_edge_d)
    return des_d, sem_edge_d


def collect_result(result):
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
    # TODO find a pretty way to use global variable
    global result_l
    for idx in range(survey.get_size()):
        pool.apply_async(describe_img_from_survey, args=(args, survey, idx),
                callback=collect_result)
    pool.close()
    pool.join()

    result_l.sort(key=lambda x: x[0])
    des_l = [r for i, r in result_l]
    result_l = []
    return des_l


def retrieve_one(args, q_idx, q_img_des, db_img_des_l):
    """ """
    #print("retrieve")
    q_label_v = np.array(list(q_img_des.keys()))
    db_size = len(db_img_des_l)
    #print("retrieve: db_size: %d"%db_size)
    img_distance_v = 1e4*np.ones(db_size)

    # compute distances with db imgs
    for db_idx in range(db_size):
        #print("db_idx: %d"%db_idx)
        db_img_des = db_img_des_l[db_idx] # dict of edge descriptor

        # check if db and q images have labels in common
        db_label_v = np.array(list(db_img_des.keys()))
        #print("retrieve: db_idx: %d"%db_idx, db_label_v)
        matching_label_v = q_label_v[np.in1d(q_label_v, db_label_v)]
        if matching_label_v.size != 0: # yes, they do
            img_distance_v[db_idx] = 0. # init img distance
        else: # no, so let the image distance to 1e4
            #print("boo")
            continue

        # image distance
        curve_count = 0
        for label in matching_label_v:
            # get the list of edge descriptor in each class in common
            q_edge_des_l = q_img_des[label]
            db_edge_des_l = db_img_des[label]
            
            # compute the distance between all pairs of edges of the same label
            q_des_v, db_des_v = np.array(q_edge_des_l), np.array(db_edge_des_l)
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
    #print("retrieve q_idx: %d"%q_idx)
    return q_idx, order



def retrieve_parallel(args, q_img_des_l, db_img_des_l):
    """ """
    q_size = len(q_img_des_l)
    pool = mp.Pool(mp.cpu_count())
    global result_l
    for q_idx in range(q_size):
        #print("1. q_idx: %d"%q_idx)
        q_img_des = q_img_des_l[q_idx]
        #toto = pool.apply_async(retrieve_one, 
        pool.apply_async(retrieve_one, 
                args=(args, q_idx, q_img_des, db_img_des_l),
                callback=collect_result)
        #toto.get()
    pool.close()
    pool.join()
    
    
    print("len(result_l): %d"%len(result_l))
    result_l.sort(key=lambda x: x[0])
    order_l = [r for i, r in result_l]
    result_l = []
    return order_l


def bench(args, n_values):
    """Runs retrieval.

    Args:
        args: retrieval parameters
        retrieval: retrieval instance
        n_value: values at which you compute recalls.
    """
    global_start_time = time.time()

    # check if this bench already exists
    perf_dir = 'res/wasabi/%d/perf'%args.trial
    mAP_fn = "%s/%d_c%d_%d_mAP.txt"%(perf_dir, args.slice_id, args.cam_id,
        args.survey_id)
    recalls_fn = "%s/%d_c%d_%d_rec.txt"%(perf_dir, args.slice_id, args.cam_id,
        args.survey_id)
    if os.path.exists(mAP_fn):
        return -1, -1 

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

    # retrieval instance
    retrieval = datasets.retrieval.Retrieval(db_survey, q_survey, args.dist_pos)
    q_survey = retrieval.get_q_survey()

    # describe db img
    local_start_time = time.time()
    db_des_fn = '%s/%d_c%d_db.pickle'%(res_dir, args.slice_id, args.cam_id)
    if not os.path.exists(db_des_fn): # if you did not compute the db already
        print('** Compute des for database img **')
        db_img_des_l = get_img_des_parallel(args, db_survey)
        with open(db_des_fn, 'wb') as f:
            pickle.dump(db_img_des_l, f)
    else: # if you already computed it, load it from disk
        print('** Load des for database img **')
        with open(db_des_fn, 'rb') as f:
            db_img_des_l = pickle.load(f)
    duration = (time.time() - local_start_time)
    print('(END) run time: %d:%02d'%(duration/60, duration%60))


    # describe q img
    local_start_time = time.time()
    q_des_fn = '%s/%d_c%d_%d.pickle'%(res_dir, args.slice_id, args.cam_id,
            args.survey_id)
    if not os.path.exists(q_des_fn): # if you did not compute the db already
        print('\n** Compute des for query img **')
        q_img_des_l = get_img_des_parallel(args, q_survey)
        with open(q_des_fn, 'wb') as f:
            pickle.dump(q_img_des_l, f)
    else: # if you already computed it, load it from disk
        print('\n** Load des for database img **')
        with open(q_des_fn, 'rb') as f:
            q_img_des_l = pickle.load(f)
    duration = (time.time() - local_start_time)
    print('(END) run time: %d:%02d'%(duration/60, duration%60))
    

    # retrieve each query
    print('\n** Retrieve query image **')
    local_start_time = time.time()
    order_l = retrieve_parallel(args, q_img_des_l, db_img_des_l)
    duration = (time.time() - local_start_time)
    print('(END) run time %d:%02d'%(duration/60, duration%60))
   

    # compute perf
    print('\n** Compute performance **')
    local_start_time = time.time()
    rank_l = retrieval.get_retrieval_rank(order_l, args.top_k)
    
    gt_name_d = retrieval.get_gt_rank("name")
    mAP = metrics.mAP(rank_l, gt_name_d)

    gt_idx_l = retrieval.get_gt_rank("idx")
    recalls = metrics.recallN(order_l, gt_idx_l, n_values)
    
    duration = (time.time() - local_start_time)
    print('(END) run time: %d:%02d'%(duration/60, duration%60))

  
    # log
    print("\nmAP: %.3f"%mAP)
    for i, n in enumerate(n_values):
        print("Recall@%d: %.3f"%(n, recalls[i]))
    duration = (time.time() - global_start_time)
    print('Global run time retrieval: %d:%02d'%(duration/60, duration%60))
    
    # write retrieval
    order_fn = "%s/%d_c%d_%d_order.txt"%(res_dir, args.slice_id, args.cam_id,
            args.survey_id)
    rank_fn = "%s/%d_c%d_%d_rank.txt"%(res_dir, args.slice_id, args.cam_id,
            args.survey_id)
    retrieval.write_retrieval(order_l, args.top_k, order_fn, rank_fn)

    # write perf
    perf_dir = 'res/wasabi/%d/perf'%args.trial
    np.savetxt(recalls_fn, np.array(recalls))
    np.savetxt(mAP_fn, np.array([mAP]))
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

    perf_dir = "res/wasabi/%d/perf/"%args.trial
    if not os.path.exists(perf_dir):
        os.makedirs(perf_dir)

    n_values = [1, 5, 10, 20]
    bench(args, n_values)


