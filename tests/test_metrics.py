""" """
import pickle

import cv2
import numpy as np

old_dir = "/home/abenbihi/ws/tf/fuck_retrieval/res/fuck/7/retrieval/"

new_trial = 7
new_dir = "/home/abenbihi/ws/tf/wasabi/res/wasabi/%d/retrieval/"%new_trial

slice_id = 24
cam_id = 0
survey_id = 0

# test db descriptor
if (1==1):

    des_old_fn = "%s/%d_c%d_db.pickle"%(old_dir, slice_id, cam_id)
    with open(des_old_fn, 'rb') as f:
        des_old_d = pickle.load(f)
    des_old_l = des_old_d["des"]
    des_old_num = len(des_old_l)
    print("des_old_l: size: %d"%des_old_num)

    des_new_fn = "%s/%d_c%d_db.pickle"%(new_dir, slice_id, cam_id)
    with open(des_new_fn, 'rb') as f:
        des_new_l = pickle.load(f)
    des_new_num = len(des_new_l)
    print("des_new_l: size: %d"%des_new_num)

    if des_old_num != des_new_num:
        raise ValueError("des_old_num != des_new_num")

    for i in range(des_new_num):
        if i in [55, 56]:
            continue
        des_old = des_old_l[i]
        des_new = des_new_l[i]

        label_old = np.array(list(des_old.keys()))
        label_new = np.array(list(des_new.keys()))
        label_num = label_old.shape[0]
    
        # test that you have the same labels
        test_label_ok = (np.sum((label_old == label_new)) == label_num)
        test_label_ok = (np.sum((label_old == label_new)) == label_num)
        if not test_label_ok:
            print("db_idx: %d"%i)
            print("label_old: ", label_old)
            print("label_new: ", label_new)
            raise ValueError("label_old != label_new")
        #else:
        #    print("label_old == label_new")
        
        # test that the descriptor are the same for each label
        for label in label_old:
            edge_des_old_l = des_old[label]
            edge_des_new_l = des_old[label]

            edge_des_num_old = len(edge_des_old_l)
            edge_des_num_new = len(edge_des_new_l)
            test_edge_num = (edge_des_num_old == edge_des_num_new)
            if not test_edge_num:
                print("edge_des_num_old: %d"%edge_des_num_old)
                print("edge_des_num_new: %d"%edge_des_num_new)
                raise ValueError("edge_des_num_old != edge_des_num_new")
            #else:
            #    print("edge_des_num_old == edge_des_num_new")


# test retrieval diff
if (0==1):
    rank_old_fn = "%s/%d_c%d_%d_rank.txt"%(old_dir, slice_id, cam_id, survey_id)
    rank_old_l = [l.split("\n")[0] for l in open(rank_old_fn, "r").readlines() ]
    
    rank_new_fn = "%s/%d_c%d_%d_rank.txt"%(new_dir, slice_id, cam_id, survey_id)
    rank_new_l = [l.split("\n")[0] for l in open(rank_new_fn, "r").readlines() ]
    
    order_old_fn = "%s/%d_c%d_%d_order.txt"%(old_dir, slice_id, cam_id, survey_id)
    print("order_old_fn: %s"%order_old_fn)
    order_old = np.loadtxt(order_old_fn, dtype=int)
    
    order_new_fn = "%s/%d_c%d_%d_order.txt"%(new_dir, slice_id, cam_id, survey_id)
    print("order_new_fn: %s"%order_new_fn)
    order_new = np.loadtxt(order_new_fn, dtype=int)
    
    a = order_old[0,:]
    b = order_new[0,:]
    print(a)
    print(b)
    order_diff = (a==b).astype(np.uint8)
    print(order_diff)
    exit(0)
    
    order_diff = (np.sum(order_old - order_new, axis=1) > 0).astype(np.uint8)
    print(np.where(order_diff == 0))
