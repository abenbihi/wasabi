"""Test reproducibility of netvlad results."""
import numpy as np

def test():
    """ """
    old_trial = 13
    old_res_dir = "third_party/tf_land/netvlad/res/%d/retrieval/"%old_trial
    
    new_trial = 3
    new_res_dir = "res/netvlad/%d/retrieval/"%new_trial

    slice_id = 22
    cam_id = 0
    survey_id = 0

    old_db_des = np.loadtxt("%s/%d_c%d_db.txt"%(old_res_dir, slice_id, cam_id))
    new_db_des = np.loadtxt("%s/%d_c%d_db.txt"%(new_res_dir, slice_id, cam_id))

    old_db_size = old_db_des.shape[0]
    new_db_size = new_db_des.shape[0]
    if old_db_size != new_db_size:
        raise ValueError("old_db_size != new_db_size")

    for i, old_des in enumerate(old_db_des):
        new_des = new_db_des[i,:]

        diff = np.sum(np.abs(new_des - old_des))
        if diff > 1e-3:
            raise ValueError("%d: old_des != new_des"%i)



if __name__=='__main__':
    test()

