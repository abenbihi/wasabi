"""Code to generate the semantic edge images in fig1.

First set: 
    - a slice
    - a camera
    - an img index in the database
It gets the nearest matching image in each traversal and draws their semantic
edges.
"""
import cv2
import numpy as np
from tools import cst, semantic_proc
import datasets.survey
from plots import casenet_plots

class DesParams(object):
    def __init__(self):
        self.min_blob_size = 50
        self.min_contour_size = 50
        self.contour_sample_num = 64
        self.top_k = 20
        self.trial = 1
        self.min_edge_size = 50

params = DesParams()

def extract_semantic_contour(params, sem_img):
    """Trial1 on semantic edge extraction."""
    debug = (0==1)
    if debug:
        colors_v = np.loadtxt('meta/color_mat.txt').astype(np.uint8)
    
    # convert semantic img to label img
    lab_img = semantic_proc.col2lab(sem_img)
    lab_img_new = semantic_proc.merge_small_blobs(lab_img)
    if debug:
        sem_img_new = semantic_proc.lab2col(lab_img_new)
        cv2.imshow('sem_img_new', sem_img_new)

    # get semantic contours
    label_l = np.unique(lab_img_new)
    contour_d = {}
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
            contour_d[label] = np.vstack(big_contours_l)

    return contour_d


def contour2prob(contour_d, edge_width, shape):
    """
    Args:
        shape: (h,w)
    """
    label_present_l = list(contour_d.keys())
    edge_prob_l = []
    for label in range(cst.LABEL_NUM):
        if label not in label_present_l:
            debug_img = np.zeros(shape)
            edge_prob_l.append(np.expand_dims(debug_img, 2))
            continue
        contour_l = contour_d[label]

        debug_img = np.zeros(shape, np.uint8)
        cv2.drawContours(debug_img, contour_l, -1, 255, edge_width)
        debug_img = debug_img.astype(np.float32)/511.
        edge_prob_l.append(np.expand_dims(debug_img, 2))
        
    prob = np.stack(edge_prob_l, axis=2)
    print(prob.shape)
    prob = np.squeeze(prob)
    prob = prob/ np.expand_dims(np.sum(prob, axis=2), 2)
    return prob



MOSAIC_SIZE = (192, 256, 3)

DATA = "cmu"
edge_width = 3

if DATA == "cmu":
    meta_dir = "meta/%s/surveys"%DATA
    img_dir = "%s/datasets/Extended-CMU-Seasons/"%cst.WS_DIR
    seg_dir = "%s/tf/cross-season-segmentation/res/ext_cmu/"%cst.WS_DIR  
    
    survey_d = {}
    
    # TODO: Choose a slice, a camera and a db img idx.
    slice_id = 22
    cam_id = 0
    idx = 100 # 24
    
    survey_l = [0,1,2,3,4,5,6,7,8,9]
    survey_l = [0,4,9]
    survey_num = len(survey_l)
    
    surveyFactory = datasets.survey.SurveyFactory()
    
    meta_fn = "%s/%d/c%d_db.txt"%(meta_dir, slice_id, cam_id)
    kwargs = {"meta_fn": meta_fn, "img_dir": img_dir, "seg_dir": seg_dir}
    db_survey = surveyFactory.create("cmu", **kwargs)
    db_pose_v = db_survey.pose_v
    
    for survey_id in survey_l: 
        meta_fn = "%s/%d/c%d_%d.txt"%(meta_dir, slice_id, cam_id, survey_id)
        kwargs["meta_fn"] = meta_fn
        q_survey = surveyFactory.create("cmu", **kwargs)
        survey_d[survey_id] = q_survey
    
    # get match of this img in all surveys
    q_idx_d = {}
    for survey_id in survey_l:
        q_pose_v = survey_d[survey_id].pose_v
        d = np.linalg.norm(np.expand_dims(db_pose_v, 1) - np.expand_dims(q_pose_v, 0), ord=None, axis=2)
        match = np.argsort(d, axis=1)[:,0]
        q_idx_d[survey_id] = match[idx]
    

    q_img_l = []
    for survey_id, q_idx in q_idx_d.items():
        if (slice_id in [24, 25]) and (cam_id==1):
            img_l.append(np.zeros(MOSAIC_SIZE, np.uint8))
            q_img_l.append(np.zeros(MOSAIC_SIZE, np.uint8))
            continue
        
        img = survey_d[survey_id].get_img(q_idx)
        sem_img = survey_d[survey_id].get_semantic_img(q_idx)
        contour_d = extract_semantic_contour(params, sem_img)
        prob = contour2prob(contour_d, edge_width, img.shape[:2])

        img_h, img_w = img.shape[:2]
        hsv_class = casenet_plots.gen_hsv_class_cityscape()
        out = casenet_plots.vis_multilabel(prob, img_h, img_w, cst.LABEL_NUM, hsv_class, use_white_background=True)
        out = out[edge_width:-edge_width, edge_width:-edge_width]
        img = img[edge_width:-edge_width, edge_width:-edge_width]
        out = out.astype(np.uint8)
        cv2.imshow("out", out)
        cv2.waitKey(0)

