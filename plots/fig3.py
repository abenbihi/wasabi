""" """
import cv2
import numpy as np
from tools import cst, semantic_proc
import datasets.survey
from plots import casenet_plots

MOSAIC_SIZE = (96,128)

class DesParams(object):
    def __init__(self):
        self.min_blob_size = 200
        self.min_contour_size = 50
        self.contour_sample_num = 64
        self.top_k = 20
        self.trial = 1
        self.min_edge_size = 50
params = DesParams()


def draw(data, survey_l, idx_l, kwargs, mosaic_size):
    mosaic_size_cv = (mosaic_size[1], mosaic_size[0])
    meta_dir = "meta/%s/surveys"%data
    col_img_l = [] # column img list
    col_sem_l = [] # colums semantic img list
    for j, (slice_id, cam_id) in enumerate(zip(slice_v, cam_v)):
        idx = idx_l[j]

        surveyFactory = datasets.survey.SurveyFactory()
        meta_fn = "%s/%d/c%d_db.txt"%(meta_dir, slice_id, cam_id)
        kwargs["meta_fn"] = meta_fn
        db_survey = surveyFactory.create(data, **kwargs)
        db_pose_v = db_survey.pose_v
        
        survey_d = {}
        for survey_id in survey_l: 
            meta_fn = "%s/%d/c%d_%d.txt"%(meta_dir, slice_id, cam_id, survey_id)
            kwargs["meta_fn"] = meta_fn
            q_survey = surveyFactory.create(data, **kwargs)
            survey_d[survey_id] = q_survey

        # get match of this img in all surveys
        q_idx_d = {}
        for survey_id in survey_l:
            q_pose_v = survey_d[survey_id].pose_v
            d = np.linalg.norm(np.expand_dims(db_pose_v, 1) - np.expand_dims(q_pose_v, 0), ord=None, axis=2)
            match = np.argsort(d, axis=1)[:,0]
            q_idx_d[survey_id] = match[idx]


        img_l = []
        sem_l = []
        for survey_id, q_idx in q_idx_d.items():
            if q_idx == -1:
                img_l.append(np.zeros((mosaic_size + (3,)), np.uint8))
                sem_l.append(np.zeros((mosaic_size + (3,)), np.uint8))
                continue
            img = survey_d[survey_id].get_img(q_idx)
            img = cv2.resize(img, mosaic_size_cv, interpolation=cv2.INTER_AREA)
            img_l.append(img)

            sem = survey_d[survey_id].get_semantic_img(q_idx)
            sem = cv2.resize(sem, mosaic_size_cv, interpolation=cv2.INTER_NEAREST)
            sem_l.append(sem)

        col_img_l.append(np.vstack(img_l))
        col_sem_l.append(np.vstack(sem_l))
   
    out_col = np.hstack(col_img_l)
    out_sem = np.hstack(col_sem_l)

    h, w = mosaic_size
    for j,_ in enumerate(col_img_l):
        out_col[:,j*w] = 255
        out_sem[:,j*w] = 255

    for i,_ in enumerate(survey_l):
        out_col[i*h, :] = 255
        out_sem[i*h, :] = 255

    cv2.imshow("out_col", out_col)
    cv2.imshow("out_sem", out_sem)
    cv2.waitKey(0)

    out = np.vstack((out_col, out_sem))
    cv2.imwrite("plots/fig3_%s.png"%data, out)


if __name__=='__main__':
    w, h = 256, 192
    DATA = "cmu"
    
    #idx_d = {}
    #idx_d[0] = [64, 100, 20, 60]
    #idx_d[1] = [100, 120, 150, 150]
 
    
    DATA = "cmu"
    meta_dir = "meta/%s/surveys"%DATA
    #DATA = "cmu"
    if DATA == "cmu":
        img_dir = "%s/datasets/Extended-CMU-Seasons/"%cst.WS_DIR
        seg_dir = "%s/tf/cross-season-segmentation/res/ext_cmu/"%cst.WS_DIR

        slice_v = np.array([22, 23, 24, 25])
        cam_v = np.array([0, 1])
        idx_l = [64, 100, 100, 120,  20, 150, 60, 150]
        survey_l = [0,2,5,6,7]
               
        #meta_fn = ""%s/%d/c%d_db.txt"%(meta_dir, slice_id, cam_id)
        kwargs = {"img_dir": img_dir, "seg_dir": seg_dir}

    elif DATA == "symphony":
        img_dir = "/mnt/lake/VBags/"
        seg_dir = "%s/tf/cross-season-segmentation/res/icra_retrieval/"%cst.WS_DIR
        mask_dir = "%s/datasets/lake/datasets/icra_retrieval/water/global"%cst.WS_DIR
               
        slice_id = 0
        cam_id = 0
        # TODO
        db_idx = 61
        survey_l = [0,4,9]

        meta_fn = "%s/%d/c%d_db.txt"%(meta_dir, slice_id, cam_id)
        kwargs = {"meta_fn": meta_fn, "img_dir": img_dir, "seg_dir": seg_dir,
                "mask_dir": mask_dir}
    else:
        raise ValueError("Unknown dataset: %s"%DATA)

    survey_num = len(survey_l)
    slice_num = slice_v.shape[0]
    slice_v = np.tile(slice_v.reshape(slice_num,1), (1,2)).flatten()
    cam_v = np.tile(cam_v, (slice_num))


    draw(DATA, survey_l, idx_l, kwargs, MOSAIC_SIZE)

