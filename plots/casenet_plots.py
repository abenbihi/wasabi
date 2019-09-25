#Copyright (c) 2017 Mitsubishi Electric Research Laboratories (MERL).   All rights reserved.
#
#The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications.  MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose.  In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.
#
#As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes. 

#visualize CASENet (Simple, fast)
# python visualize_multilabel.py ~/datasets/Cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png
#visualize CASENet (Full, slow)
# python visualize_multilabel.py ~/datasets/Cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png -g ~/datasets/Cityscapes/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_edge.bin
import os
import numpy as np
import numpy.random as npr
import cv2
import struct
import sys
import time
import multiprocessing
import argparse

#NEW_H = 384 #1024 # Need to pre-determine test image size
#NEW_W = 1024 #2048 # Need to pre-determine test image size


def gen_hsv_class_cityscape():
    return np.array([
        359, # 0.road
        320, # 1.sidewalk
        40,  # 2.building
        80,  # 3.wall
        90,  # 4.fence
        10,  # 5.pole
        20,  # 6.traffic light
        30,  # 7.traffic sign
        140, # 8.vegetation
        340, # 9.terrain
        280, # 10.sky
        330, # 11.person
        350, # 12.rider
        120, # 13.car
        110, # 14.truck
        130, # 15.bus
        150, # 16.train
        160, # 17.motorcycle
        170  # 18.bike
    ])/2.0


def get_class_name_cityscape():
    return ['road',
            'sidewalk',
            'building',
            'wall',
            'fence',
            'pole',
            'traffic light',
            'traffic sign',
            'vegetation',
            'terrain',
            'sky',
            'person',
            'rider',
            'car',
            'truck',
            'bus',
            'train',
            'motorcycle',
            'bicycle']


#def gen_hsv_class(K_class, h_min=1, h_max=179, do_shuffle=False, do_random=False):
#    hsv_class = np.linspace(h_min, h_max, K_class, False, dtype=np.int32)
#    if do_shuffle:
#        npr.shuffle(hsv_class)
#    if do_random:
#        hsv_class = npr.randint(h_min, h_max, K_class, dtype=np.int32)
#    return hsv_class


#def save_each_blend(fname_prefix, prob, thresh=0.5, names=None):
#    """
#    My version of coloring mono class map
#    """
#
#    out_dir = fname_prefix.split(".")[0]
#    if not os.path.exists(out_dir):
#        os.makedirs(out_dir)
#
#    if names is None:
#        names = get_class_name_cityscape()
#    rows,cols,nchns = prob.shape
#    H_TP = 60.0
#    H_FN = 120.0
#    use_abs_color = True
#    hsv_class = gen_hsv_class(19)
#    i_color = 0
#
#    # H_FP_OR_TN = 0
#    #class_to_keep = [ 0, 1, 7, 8, 10, 13]
#    class_to_keep = list(range(19)) #[ 0, 1, 8, 10, 11, 12, 18, 13]
#
#    img_to_keep = []
#    for k in range(0, nchns):
#        #print(k)
#        hi = hsv_class[ k if use_abs_color else i_color ]
#        namek = fname_prefix + names[k] + '.png'
#        probk = prob[:,:,k]
#        #probk_gt = prob_gt[:,:,k]
#
#        #label_pos = probk_gt>0
#        #label_neg = probk_gt<=0
#        pred_pos = probk>thresh
#        pred_neg = probk<=thresh
#        label_pos = pred_pos
#        label_neg = pred_neg
#        #label_pos = np.ones((rows, cols))
#
#        blendk_hsv = np.zeros((rows,cols,3), dtype=np.float32)
#        #blendk_hsv[:,:,0] = hi*label_pos*(pred_pos*H_TP + pred_neg*H_FN) #classification result type
#        #blendk_hsv[:,:,1] = label_pos*(pred_pos*probk*255.0 + pred_neg*(1-probk)*255.0) + label_neg*probk*255.0 #probability
#        
#        blendk_hsv[:,:,0] = hi*pred_pos #classification result type
#        blendk_hsv[:,:,1] = pred_pos*probk*255.0 
#
#        blendk_hsv[:,:,2] = 255
#        blend = cv2.cvtColor(blendk_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
#        if k in class_to_keep:
#            img_to_keep.append(blend.astype(np.uint8))
#    return blend
#        #out_fn = os.path.join(out_dir, '%d.png'%k)
#        #cv2.imwrite(out_fn, blend.astype(np.uint8))
#        #print(names[k])
#        #cv2.imshow('probk', probk)
#        #cv2.imshow('blend', blend)
#        #cv2.waitKey(0)
#    
#
#    #res = np.hstack(img_to_keep)
#    #for i in range(1,len(class_to_keep)):
#    #    res[:,i*cols: i*cols+1,:] = 0
#    #cv2.imshow('res', res)
#    #cv2.imwrite(fname_prefix, res)
#    #print(fname_prefix)
#    #cv2.waitKey(0)
#
#        #cv2.imwrite(namek, blend.astype(np.uint8))
#
#
#def save_each_blended_probk(fname_prefix, prob_gt, prob, thresh=0.5, names=None):
#    if names is None:
#        names = get_class_name_cityscape()
#    rows,cols,nchns = prob.shape
#    H_TP = 60.0
#    H_FN = 120.0
#    # H_FP_OR_TN = 0
#    for k in range(0, nchns):
#        namek = fname_prefix + names[k] + '.png'
#        probk = prob[:,:,k]
#        probk_gt = prob_gt[:,:,k]
#
#        label_pos = probk_gt>0
#        label_neg = probk_gt<=0
#        pred_pos = probk>thresh
#        pred_neg = probk<=thresh
#
#        blendk_hsv = np.zeros((rows,cols,3), dtype=np.float32)
#        blendk_hsv[:,:,0] = label_pos*(pred_pos*H_TP + pred_neg*H_FN) #classification result type
#        blendk_hsv[:,:,1] = label_pos*(pred_pos*probk*255.0 + pred_neg*(1-probk)*255.0) + label_neg*probk*255.0 #probability
#        blendk_hsv[:,:,2] = 255
#        blend = cv2.cvtColor(blendk_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
#
#        cv2.imwrite(namek, blend.astype(np.uint8))
#
#
#def thresh_and_select_k_largest_only(prob, k=2, thresh=0.0):
#    prob[prob <= thresh] = 0
#
#    rows = prob.shape[0]
#    cols = prob.shape[1]
#    chns = prob.shape[2]
#
#    if k<=0 or k>=chns:
#        return prob
#
#    ii, jj = np.meshgrid(range(0,rows), range(0,cols), indexing='ij')
#
#    prob_out = np.zeros(prob.shape, dtype=np.float32)
#    for ik in range(0, k):
#        idx_max = np.argmax(prob, axis=2)
#        prob_out[ii, jj, idx_max] = prob[ii, jj, idx_max]
#        prob[ii, jj, idx_max] = -1
#    return prob_out


def vis_multilabel(prob, img_h, img_w, K_class, hsv_class=None, use_white_background=False):
    label_hsv = np.zeros((img_h, img_w, 3), dtype=np.float32)
    prob_sum = np.zeros((img_h, img_w), dtype=np.float32)
    prob_edge = np.zeros((img_h, img_w), dtype=np.float32)

    use_abs_color = True
    #if hsv_class is None:
    #    n_colors = 0
    #    use_abs_color = False
    #    for k in range(0, K_class):
    #        if prob[:, :, k].max() > 0:
    #            n_colors += 1
    #    #hsv_class = gen_hsv_class(n_colors)
    #    hsv_class = gen_hsv_class(K_class)
    #    print( hsv_class)

    i_color = 0
    for k in range(0, K_class):
        prob_k = prob[:, :, k].astype(np.float32)
        if prob_k.max() == 0:
            continue
        hi = hsv_class[ k if use_abs_color else i_color ]
        i_color += 1
        label_hsv[:, :, 0] += prob_k * hi  # H
        prob_sum += prob_k
        prob_edge = np.maximum(prob_edge, prob_k)

    prob_sum[prob_sum == 0] = 1.0
    label_hsv[:, :, 0] /= prob_sum
    if use_white_background:
        label_hsv[:, :, 1] = prob_edge * 255
        label_hsv[:, :, 2] = 255
    else:
        label_hsv[:, :, 1] = 255
        label_hsv[:, :, 2] = prob_edge * 255

    label_bgr = cv2.cvtColor(label_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return label_bgr


#def bits2array(vals):
#    # print vals
#    return [[(valj >> k) & 1
#             for k in range(0, 32)]
#            for valj in vals]
#
#
#def load_gt(fname_gt, img_h, img_w, use_inner_pool=False):
#    # timer = Timer(1000)
#    # timer.tic()
#    with open(fname_gt, 'rb') as fp:
#        bytes = fp.read(img_h * img_w * 4)
#    # timer.toc(tag='[load_gt.open]')
#
#    # timer.tic()
#    labeli = np.array(struct.unpack('%dI' % (len(bytes) / 4), bytes)).reshape((img_h, img_w))
#    # timer.toc(tag='[load_gt.unpack]')
#
#    #find unique labels
#    # timer.tic()
#    labeli_unique, labeli_count = np.unique(labeli, return_counts=True)
#    sidx = np.argsort(labeli_count)[::-1]
#    labeli_unique=labeli_unique[sidx]
#    # labeli_count=labeli_count[sidx]
#    # timer.toc(tag='[load_gt.unique]')
#
#    # timer.tic()
#    nbits = 32
#
#    if use_inner_pool:
#        # prob = np.float32(map(bits2array,labeli))
#        pool = multiprocessing.Pool()
#        prob = pool.map(bits2array, labeli.tolist())
#        pool.close()
#        pool.join()
#        prob = np.float32(prob)
#    else:
#        prob = np.float32([[[(labeli[i, j] >> k) & 1
#                             for k in range(0, nbits)]
#                            for j in range(0, img_w)]
#                           for i in range(0, img_h)])
#    # timer.toc(tag='[load_gt.for]')
#    return prob, labeli_unique[1:] if labeli_unique[0]==0 else labeli_unique
#
#
#def load_result(img_h, img_w, result_fmt, K_class, idx_base):
#    prob = np.zeros((img_h, img_w, K_class), dtype=np.float32)
#    for k in range(K_class):
#        print(result_fmt%(k+idx_base))
#        prob_fn = result_fmt%(k+idx_base)
#        prob_fn = '%s.png'%prob_fn.split(".")[0]
#        prob_k = cv2.imread(prob_fn, cv2.IMREAD_GRAYSCALE)
#        prob[:,:,k] = prob_k.astype(np.float32) / 255.
#    return prob
#
#
#def main_multi(out_name=None, raw_name=None, gt_name=None,
#         result_fmt=None, thresh=0.5, do_gt=True, select_only_k = 2,
#         do_each_comp = False, K_class = 19, idx_base = 0, save_input=False, dryrun=False):
#    oldstdout = sys.stdout
#    if do_gt:
#        sys.stdout = open(out_name+'.log', 'w')
#
#    hsv_class = gen_hsv_class_cityscape()
#    class_names = get_class_name_cityscape()
#
#    timer = Timer(1000)
#
#    #load files
#    image = cv2.imread(raw_name)
#    # resizing
#    # get img dim for this seq
#    h,w = image.shape[:2]
#    
#    MOD = 8
#    if w%MOD!=0:
#        #pad_w = (8)*(int(w/8) + 1) - w
#        pad_w = MOD * (int(w/MOD) + 1) - w
#        #print( MOD*(int(w/MOD) + 1) - w)
#        if pad_w%2==0:
#            pad_w_l, pad_w_r = pad_w/2, pad_w/2 # pad left, right
#        else:
#            pad_w_l = int(pad_w/2)
#            pad_w_r = pad_w - pad_w_l
#    else:
#        pad_w, pad_w_l, pad_w_r = 0,0,0
#    #print('pad_w: %d - pad_w_l: %d - pad_w_l: %d'%(pad_w, pad_w_l, pad_w_r))
#    
#    if h%MOD!=0:
#        pad_h = MOD * (int(h/MOD) + 1) - h
#        #print( MOD*(int(h/MOD) + 1) - h)
#        if pad_h%2==0:
#            pad_h_t, pad_h_b = pad_h/2, pad_h/2 # pad top, bottom
#        else:
#            pad_h_t = int(pad_h/2)
#            pad_h_b = pad_h - pad_h_t
#    else:
#        pad_h, pad_h_t, pad_h_b = 0,0,0
#    #print('pad_h: %d - pad_h_l: %d - pad_h_l: %d'%(pad_h, pad_h_t, pad_h_b))
#    
#    image_h = h + pad_h #376 
#    image_w = w + pad_w # 1248 
#    #print('image_w: %d - image_h: %d'%(image_w, image_h))
#
#    image = cv2.copyMakeBorder(image, pad_h_t, pad_h_b, pad_w_l, pad_w_r, cv2.BORDER_REFLECT)
#    #image = image[:, :NEW_W,:]
#    #image = cv2.resize(image, (NEW_W, NEW_H), interpolation=cv2.INTER_AREA) 
#
#    #print(raw_name)
#    img_h, img_w = (image.shape[0], image.shape[1])
#    if do_gt:
#        # timer.tic()
#        prob_gt, labeli = load_gt(gt_name, img_h, img_w)
#        # timer.toc(tag='[load_gt]')
#        output_color_definition_for_latex(hsv_class, labeli,names=class_names)
#    prob = load_result(img_h, img_w, result_fmt, K_class, idx_base)
#
#    # vis gt and blended_gt
#    if do_gt:
#        # timer.tic()
#        label_bgr = vis_multilabel(prob_gt, img_h, img_w, K_class, hsv_class, use_white_background=True)
#        # timer.toc(tag='[vis_multilabel for gt]')
#        if dryrun:
#            print( 'writing: '+out_name+'.gt.png')
#        else:
#            cv2.imwrite(out_name+'.gt.png', label_bgr)
#    if save_input:
#        if dryrun:
#            print('writing: '+out_name+'.input.png')
#        else:
#            cv2.imwrite(out_name+'.input.png', image)
#        # blended = cv2.addWeighted(image, 0.2, label_bgr, 0.8, 0)
#        # cv2.imwrite(fout_prefix+'_blend_gt.png', blended)
#
#    # # vis result
#    if select_only_k>0 or thresh>0.0:
#        # timer.tic()
#        prob_out = thresh_and_select_k_largest_only(prob, select_only_k, thresh)
#        prob = prob_out
#        # timer.toc(tag='[thresh_and_select_k_largest_only]')
#    if do_each_comp and do_gt:
#        # timer.tic()
#        save_each_blended_probk(out_name+'.comp.',prob_gt,prob,names=class_names)
#        # timer.toc(tag='[save_each_blended_probk]')
#    label_bgr = vis_multilabel(prob, img_h, img_w, K_class, hsv_class, use_white_background=True)
#    if dryrun:
#        print( 'writing: '+out_name)
#    else:
#        #res = np.vstack((image, label_bgr))
#        #res = cv2.resize(res, (int(res.shape[1]/2), int(res.shape[0]/2)),
#        #        interpolation=cv2.INTER_AREA)
#        #cv2.imwrite(out_name, res)
#        cv2.imwrite(out_name, label_bgr)
#
#    #print 'finished: '+raw_name
#    sys.stdout = oldstdout
#
#
#def resize_mirror(image):
#    h,w = image.shape[:2]
#    
#    MOD = 8
#    if w%MOD!=0:
#        #pad_w = (8)*(int(w/8) + 1) - w
#        pad_w = MOD * (int(w/MOD) + 1) - w
#        #print( MOD*(int(w/MOD) + 1) - w)
#        if pad_w%2==0:
#            pad_w_l, pad_w_r = pad_w/2, pad_w/2 # pad left, right
#        else:
#            pad_w_l = int(pad_w/2)
#            pad_w_r = pad_w - pad_w_l
#    else:
#        pad_w, pad_w_l, pad_w_r = 0,0,0
#    #print('pad_w: %d - pad_w_l: %d - pad_w_l: %d'%(pad_w, pad_w_l, pad_w_r))
#    
#    if h%MOD!=0:
#        pad_h = MOD * (int(h/MOD) + 1) - h
#        #print( MOD*(int(h/MOD) + 1) - h)
#        if pad_h%2==0:
#            pad_h_t, pad_h_b = pad_h/2, pad_h/2 # pad top, bottom
#        else:
#            pad_h_t = int(pad_h/2)
#            pad_h_b = pad_h - pad_h_t
#    else:
#        pad_h, pad_h_t, pad_h_b = 0,0,0
#    #print('pad_h: %d - pad_h_l: %d - pad_h_l: %d'%(pad_h, pad_h_t, pad_h_b))
#    
#    image_h = h + pad_h #376 
#    image_w = w + pad_w # 1248 
#    #print('image_w: %d - image_h: %d'%(image_w, image_h))
#
#    image = cv2.copyMakeBorder(image, pad_h_t, pad_h_b, pad_w_l, pad_w_r, cv2.BORDER_REFLECT)
#    return image
#
#
#def main_mono(out_name=None, raw_name=None, gt_name=None,
#         result_fmt=None, thresh=0.5, do_gt=True, select_only_k = 2,
#         do_each_comp = False, K_class = 19, idx_base = 0, save_input=False, dryrun=False):
#    oldstdout = sys.stdout
#    if do_gt:
#        sys.stdout = open(out_name+'.log', 'w')
#
#    hsv_class = gen_hsv_class_cityscape()
#    class_names = get_class_name_cityscape()
#
#    timer = Timer(1000)
#
#    #load files
#    image = cv2.imread(raw_name)
#    image = resize_mirror(image)
#
#    #print(raw_name)
#    img_h, img_w = (image.shape[0], image.shape[1])
#    prob = load_result(img_h, img_w, result_fmt, K_class, idx_base)
#
#    # # vis result
#    if select_only_k>0 or thresh>0.0:
#        prob_out = thresh_and_select_k_largest_only(prob, select_only_k, thresh)
#        prob = prob_out
#    save_each_blend(out_name, prob, names=class_names)
#    
#
#    #label_bgr = vis_multilabel(prob, img_h, img_w, K_class, hsv_class, use_white_background=True)
#    #if dryrun:
#    #    print 'writing: '+out_name
#    #else:
#    #    cv2.imwrite(out_name, label_bgr)
#
#    #print 'finished: '+raw_name
#    #sys.stdout = oldstdout
#
#
#def get_color_for_latex():
#    hsv_class = gen_hsv_class_cityscape()
#    class_names = get_class_name_cityscape()
#
#    output_color_definition_for_latex(hsv_class, None,names=class_names)

