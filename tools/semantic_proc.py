"""Set of primitives to process semantic maps and colors."""
import os
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

from tools import cst
from tools import edge_proc

def col2lab(col, colors=cst.palette_bgr):
    """Convert color map to label map.
    WARNING: use cst.palette_bgr from the cityscapes palette and assumes the colors
    are ordered to match their label.

    Args:
        col: (h,w,3) color semantic img.
        colors: list of bgr values ordered in the label order.

    Returns:
        lab: (h,w) label semantic img.
    """
    lab = cst.LABEL_IGNORE * np.ones(col.shape[:2]).astype(np.uint8)
    for i, color in enumerate(colors):
        # I know, this is ugly 
        mask = np.zeros(col.shape[:2]).astype(np.uint8)
        mask = 255*(col==color).astype(np.uint8)
        mask = (np.sum(mask,axis=2) == (255*3)).astype(np.uint8)
        lab[mask==1] = i
    return lab


def lab2col(lab, colors=cst.palette_bgr):
    """Convert label map to color map.
    
    Args:
        lab: (h,w) label semantic img.
        colors: list of bgr values ordered in the label order.

    Returns:
        col: (h,w,3) color semantic img.
    """
    col = np.zeros((lab.shape + (3,))).astype(np.uint8)
    labels = np.unique(lab)
    if np.max(labels) >= len(colors):
        raise ValueError("Error: you need more colors np.max(labels) >= "
                "len(colors): %d >= %d"%(np.max(labels), len(colors)) )
    for label in labels:
        col[lab==label,:] = colors[label]
    return col


def merge_small_blobs(lab):
    """Denoise label img by removing small blobs.

    Args: 
        lab: (h,w) semanti label img

    Returns:
        lab: (h,w) semanti label img
    """
    lab_copy = lab.copy()
    lab_id_l = np.unique(lab)
    rows, cols = lab.shape[:2]
    lab_to_merge = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14]
    lab_to_ignore = [4, 5, 6, 7]

    for lab_id in lab_id_l:
        if lab_id in lab_to_ignore:
            continue
        # mask all pixels with label lab_id
        l,c = np.where(lab_copy==lab_id)
        mask_lab_id = np.zeros(lab.shape).astype(np.uint8)
        mask_lab_id[l,c] = 255
        #print('Proc lab_id: %d'%lab_id)

        while len(l) != 0:
            #print(l)
            mask = np.zeros(lab.shape).astype(np.uint8)
            mask[l,c] = 255
            # output of flood fill, 1 where it has filled
            # I don't know why, it does not want to fill with 128
            mask_out = np.zeros((lab.shape[0]+2, lab.shape[1]+2)).astype(np.uint8)
            cv2.floodFill(mask, mask_out, (c[0], l[0]), 128)
            cc_area = np.sum(mask_out[1:-1,1:-1]==1) # blob area
            
            lab_copy[mask_out[1:-1, 1:-1]==1] = -1
            l, c = np.where(lab_copy==lab_id)
            
            # if this connected components is big enough, keep it
            if cc_area > 500:
                continue
            # else merge it with its neighbour
            
            # define neighbour range
            lab[mask_out[1:-1, 1:-1]==1] = cst.LABEL_IGNORE
            r_mask, c_mask = np.where(mask_out[1:-1, 1:-1] == 1)
            patch_size = 20
            br = np.maximum(0, np.min(r_mask) - patch_size)
            er = np.minimum(rows, np.max(r_mask) + patch_size)
            bc = np.maximum(0, np.min(c_mask) - patch_size)
            ec = np.minimum(cols, np.max(c_mask) + patch_size)
            
            lab_neighbour = lab[br:er, bc:ec]

            y_db, x_db = np.where(lab_neighbour!=cst.LABEL_IGNORE)
            y_db += br
            x_db += bc
            y_q, x_q = np.where(lab==cst.LABEL_IGNORE)
            
            knn = NearestNeighbors(n_jobs=1)
            knn.fit(np.vstack((x_db, y_db)).T)
            positives = knn.kneighbors(np.vstack((x_q, y_q)).T, 1,
                    return_distance=False)
            positives = np.squeeze(positives)
            lab[y_q, x_q] = lab[y_db[positives], x_db[positives]]

    return lab


def extract_semantic_edge(params, sem_img):
    """Trial1 on semantic edge extraction."""
    debug = (0==1)
    if debug:
        colors_v = np.loadtxt('meta/color_mat.txt').astype(np.uint8)
    
    # convert semantic img to label img
    lab_img = col2lab(sem_img)
    lab_img_new = merge_small_blobs(lab_img)
    if debug:
        sem_img_new = lab2col(lab_img_new)
        cv2.imshow('sem_img_new', sem_img_new)

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

    # extract connected components from the contour
    curves_d = {}
    for label, contour in contours_d.items():
        curves = edge_proc.contours2curve(params, contour, sem_img)
        if curves is not None:
            curves_d[label] = curves
    return curves_d
