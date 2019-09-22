"""
Extract all_curves from contour.
"""
import os

import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

class DesParams(object):
    def __init__(self):
        self.min_blob_size = 300
        self.min_contour_size = 512
        self.contour_sample_num = 64


def contour2patch(contour, patch_shape, color=255):
    """Draw normalised contour on patch."""
    patch = np.zeros(patch_shape, np.uint8)
    contour = np.squeeze(contour).astype(np.float32)

    mean_x, mean_y = np.mean(contour, axis=0)
    #contour[:,0] -= mean_x
    #contour[:,1] -= mean_y
    old_min = np.min(contour)
    old_max = np.max(contour)

    border = 20
    new_min = border
    new_max = patch_shape[0] - border
    contour = new_min + (contour - old_min) * (new_max - new_min) / (old_max -
            old_min)
    #print('X\tmin: %.1f\tmax: %.1f'%( np.min(contour[:,0]),
    #    np.max(contour[:,0])) )
    #print('Y\tmin: %.1f\tmax: %.1f\n'%( np.min(contour[:,1]),
    #    np.max(contour[:,1])) )

    contour = contour.astype(np.int)
    
    # use draw contour
    patch = np.zeros(patch_shape[:2], np.uint8)
    contour = np.expand_dims(contour, 1)
    #print(patch.shape)
    cv2.drawContours(patch, [contour], 0, 255, 1)
    if len(patch_shape) == 3:
        l, c = np.where(patch==255)
        patch = np.tile(np.expand_dims(patch, 2), (1,1,3))
        patch[l,c,:] = color

    #patch = np.zeros(patch_shape, np.uint8)
    #patch[contour[:,1], contour[:,0]] = color

    #cv2.imshow('patch', patch)
    #cv2.waitKey(0)

    contour = np.squeeze(contour)
    contour = np.unique(contour, axis=0)

    return contour, patch


#def gen_data(contours_l, patch_shape, color=255):
def gen_patches(contours_l, patch_shape, color=255):
    """Normalises contours and draws them on individual patches."""
    c_l, c_img_l = [], []
    for contour in contours_l:
        c, c_img = contour2patch(contour, patch_shape, color)
        c_l.append(c)
        c_img_l.append(c_img)

    return c_l, c_img_l


def fuse_patches(c_img_l):
    """Draws contour patches next to each other."""
    patch_num = len(c_img_l)
    patch_h, patch_w = c_img_l[0].shape[:2]
    out = np.hstack(c_img_l)
    for i in range(1, patch_num):
        out[:, i*patch_w] = 255
    return out


def get_edges(img):
    """Extracts edged with Canny."""
    if img.ndim == 3:
        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_bw = img
    # edges
    k = 3
    blur = cv2.blur(img_bw, (k,k))
    lowThreshold = 50
    ratio = 5
    edges = cv2.Canny(blur, lowThreshold, ratio*lowThreshold, 3)
    return edges


def get_edge_density(img):
    """Computes heatmap of edges."""
    edges = get_edges(img)
    # edge density
    k = 15
    density = cv2.blur(edges, (k,k))
    return edges, density


def contours2curve(params, contour, q_img):
    curves_l = []
    debug = (0==1)
    eps = 10
    
    # split contour in useful all_curves i.e. not border img
    all_curve = contour.copy()
    left = (all_curve[:,:,0] == 0)
    right = (all_curve[:,:,0] == q_img.shape[1]-1)
    top = (all_curve[:,:,1] == 0)
    bottom = (all_curve[:,:,1] == q_img.shape[0]-1)
    ok = (~left) & (~right) & (~top) & (~bottom)
    ok = np.squeeze(ok)
    all_curve = all_curve[ok==1,:,:]
    all_curve = np.squeeze(all_curve)
    c_num = all_curve.shape[0]
    #print('c_num: %d'%c_num)

    if c_num < params.min_edge_size:
        return None

    d = np.linalg.norm( np.expand_dims(all_curve, 1) -
            np.expand_dims(all_curve, 0), ord=None, axis=2)
    mask_neighbour = d<eps
    
    if debug:
        contour_img = np.zeros(q_img.shape[:2], np.uint8)
        cv2.drawContours(contour_img, [contour], 0, 128, 3)
        cv2.imshow('contour_img', contour_img)
        cv2.waitKey(1)

    
    head = 0
    while np.sum(mask_neighbour) != 0:
        queue = []
        # init current curve
        while head < (c_num-1):
            free_neighbour_idx = np.where(mask_neighbour[head, :] == 1)[0]
            if free_neighbour_idx.size == 0:
                #print("Pas de chance, this pt has no neighbour")
                head += 1
                continue
            else:
                break
        if head == c_num-1:
            mask_neighbour[:,head] = 0
            #print("Error: head == c_num-1")
            #exit(1)
        #input('wait')
        #print(free_neighbour_idx)
        #input('free_neighbour_idx')

        if free_neighbour_idx.size == 0:
            break
    
        mask_neighbour[:,free_neighbour_idx] = 0
        queue += list(free_neighbour_idx)
        current_curve = all_curve[free_neighbour_idx,:]
        
        #print(d.shape)
        #print(head)
        #print(free_neighbour_idx)
        #print(free_neighbour_idx.size)
        #print(np.max(free_neighbour_idx))
        max_dist = np.max(d[head,free_neighbour_idx])
        
        if debug:
            print('head: %d'%head)
            print('max_dist: %.3f'%max_dist)
            #print(current_curve)
            #input('current_curve')
            #print(queue)
            #input('queue')
            curve_img = np.zeros(q_img.shape[:2], np.uint8)
            contour_img[current_curve[:,1], current_curve[:,0]] = 255
            cv2.imshow('contour_img', contour_img)
            if (cv2.waitKey(0) & 0xFF) == ord("q"):
                exit(0)
        
        while len(queue)!=0:
            head = queue.pop(0)
            #print('\nhead: %d'%head)
            #if np.sum(mask_neighbour[:, head]) == 0:
            #    input('Already added')
            #    continue # already added
            #else:
            free_neighbour_idx = np.where(mask_neighbour[head, :] == 1)[0]
            #print('free_neighbour_idx')
            if free_neighbour_idx.size == 0:
                #print('this point has no neighbour')
                continue # this pt has no neighbour
            else:
                mask_neighbour[:,free_neighbour_idx] = 0
                queue += list(free_neighbour_idx)
                current_curve = np.vstack((current_curve,
                    all_curve[free_neighbour_idx,:]))
        
        curves_l.append(current_curve)
        
        if debug:
            # at this point you have one curve
            curve_img = np.zeros(q_img.shape[:2], np.uint8)
            contour_img[current_curve[:,1], current_curve[:,0]] = 255
            cv2.imshow('curve_img', contour_img)
            if (cv2.waitKey(0) & 0xFF) == ord("q"):
                exit(0)
    
    # sort the points in the curves so that the curves are continuous (yes,
    # this is heavy)
    sorted_curves_l = []
    for curve in curves_l:
        if curve.shape[0] < 10: # discard zigouigoui
            continue
        curve_img = np.zeros(q_img.shape[:2], np.uint8)
        curve_img[curve[:,1], curve[:,0]] = 255

        im2, contours, hierarchy = cv2.findContours(curve_img,
                cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        contour_idx = 0
        if len(contours) > 1:
            # keep the longest one
            max_size = -1
            contour_idx = -1
            for i, c in enumerate(contours):
                if c.shape[0] > max_size:
                    contour_idx = i
                    max_size = c.shape[0]
            #cv2.drawContours(curve_img, contours, 0, 128, 3) 
            #print(contours[0])
            #print(contours[1])
            #print(len(contours))
            #curve_img[curve[:,1], curve[:,0]] = 255
            #cv2.imshow('ko_curve_img', curve_img)
            #if (cv2.waitKey(0) & 0xFF) == ord("q"):
            #    exit(0)           
        sorted_curves_l.append(np.squeeze(contours[contour_idx]))

    return sorted_curves_l


#def test_contours2curve():
#    """  """
#    slice_id, cam_id, survey_id = 24, 0, 1
#    eps = 5
#    params = DesParams()
#
#    
#    survey = Survey.Survey(slice_id, cam_id, survey_id)
#
#    for q_idx in range(survey.img_num):
#        q_img = survey.get_img(q_idx)
#        q_sem_img = survey.get_sem_img(q_idx)
#        
#        q_contours_l = tools_sem.gimme_gimme_gimme_a_contour(params, q_sem_img)
#        
#        curves_l = []
#        contour_img = np.zeros(q_img.shape[:2], np.uint8)
#        for contour in q_contours_l:
#            cv2.drawContours(contour_img, [contour], 0, 128, 3)
#            cv2.imshow('contour_img', contour_img)
#            cv2.waitKey(1)
#            #skip = cv2.waitKey(0) & 0xFF
#            #if skip == ord("s"):
#            #    continue
#
#            curves_l += contours2curve(contour, q_img)
#        
#        des_l = []
#        for i, curve in enumerate(curves_l):
#            des_l.append(tools_des_edge.wavelet_chuang96(params, curve, None))
#
#        
#        plt.figure()
#        absc = np.arange(2*params.contour_sample_num)
#        for i, des in enumerate(des_l):
#            plt.plot(absc, des, label='%d'%i)
#        plt.legend()
#        plt.savefig('trash/toto.png')
#        plt.close()
#        toto = cv2.imread('trash/toto.png')
#        cv2.imshow('toto', toto)
#        cv2.waitKey(1)
#
#        
#        curve_img_d = {}
#        for i, curve in enumerate(curves_l):
#            curve_img = np.zeros(q_img.shape[:2], np.uint8)
#            curve_img[curve[:,1], curve[:,0]] = 255
#
#            #im2, contours, hierarchy = cv2.findContours(curve_img,
#            #        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
#            ##print(contours)
#            ## my extraction does not sort the points and this function does
#            ## I love you opencv
#            #cv2.drawContours(curve_img, contours, 0, 128, 3) 
#            
#            curve_img_d[i] = curve_img
#            #curve_img[curve[:,1], curve[:,0]] = 255
#            cv2.imshow('%d curve_img'%i, curve_img)
#        if (cv2.waitKey(0) & 0xFF) == ord("q"):
#            exit(0)
#
#        
#        #out = curve_img_d[1] + curve_img_d[3]
#        #out[out>255] = 128
#        #cv2.imshow('out', out)
#        #cv2.waitKey(0)
#
#if __name__=='__main__':
#    test_contours2curve()

