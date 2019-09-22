"""Set of primitives to process semantic maps."""
import os
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import tools.cst as cst

def col2lab(col, colors=cst.palette_bgr):
    """
    Convert color map to label map.
    WARNING: use palette_bgr from the cityscapes palette and assumes the colors
    are ordered to match their label.
    """
    #kernel = np.ones((2,2),np.uint8)
    #col = cv2.dilate(col, kernel,iterations = 1)

    lab = label_ignore * np.ones(col.shape[:2]).astype(np.uint8)
    
    #pixel_num = 0
    #print(col)
    #print(col.dtype)

    #cv2.imshow('col', col)
    #cv2.waitKey(0)

    for i, color in enumerate(colors):
        #print((col==color).astype(np.uint8))
        #print(i)
        #print('%d: %s'%(i, palette.label_name[i]), color)
        # I know, this is ugly 
        mask = np.zeros(col.shape[:2]).astype(np.uint8)
        mask = 255*(col==color).astype(np.uint8)
        mask = (np.sum(mask,axis=2) == (255*3)).astype(np.uint8)
        lab[mask==1] = i

        #col[mask==1] = 0
        #pixel_num += np.sum(mask)
        #cv2.imshow('lab', lab)
        #cv2.imshow('mask', mask)

    #print(np.unique(col[:,:,0]))
    #cv2.imshow('col', col)
    #stop_show = cv2.waitKey(0) & 0xFF
    #if stop_show == ord("q"):
    #    exit(0)

    #print(np.where(lab==100))
    #h,w = col.shape[:2]
    #input('pixel_num / h*w: %d / %d'%(pixel_num, h*w))

    return lab


def lab2col(lab, colors=cst.palette_bgr):
    """
    Convert label map to color map
    """
    col = np.zeros((lab.shape + (3,))).astype(np.uint8)
    labels = np.unique(lab)

    if np.max(labels) >= len(colors):
        print("Error: you need more colors np.max(labels) >= len(colors): %d >= %d"
                %(np.max(labels), len(colors)) )
        exit(1)

    for label in labels:
        col[lab==label,:] = colors[label]
    return col


def merge_small_blobs(lab):
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
            
            # debug: to get a visual idea of a blob size
            #print('cc_id: %d\tsize: %d\treal size: %d'%(next_cc_id-1, len(l), cc_area))
            #cv2.imshow('current cc', (mask_out[1:-1,1:-1]*255).astype(np.uint8))
            #stop_show = cv2.waitKey(0) & 0xFF
            #if stop_show == ord("q"):
            #    exit(0)
            
            lab_copy[mask_out[1:-1, 1:-1]==1] = -1
            l, c = np.where(lab_copy==lab_id)
            
            # if this connected components is big enough, keep it
            if cc_area > 500:
                #print('big area')
                continue
            #print('blob_size: %d'%cc_area)

            # else merge it with its neighbour
            
            # define neighbour range
            lab[mask_out[1:-1, 1:-1]==1] = label_ignore
            r_mask, c_mask = np.where(mask_out[1:-1, 1:-1] == 1)
            patch_size = 20
            br = np.maximum(0, np.min(r_mask) - patch_size)
            er = np.minimum(rows, np.max(r_mask) + patch_size)
            bc = np.maximum(0, np.min(c_mask) - patch_size)
            ec = np.minimum(cols, np.max(c_mask) + patch_size)
            
            #cv2.rectangle(lab, (bc, br), (ec, er), 255, 1, lineType=8)
            #cv2.imshow('lab', lab)
            #cv2.waitKey(0)
            #cv2.rectangle(lab, (bc, br), (ec, er), 0, 1, lineType=8)
            #print(br, er, bc, ec)
            
            lab_neighbour = lab[br:er, bc:ec]
            #print(np.unique(lab_neighbour))

            y_db, x_db = np.where(lab_neighbour!=label_ignore)
            #print(y_db)
            #print(br, er)
            #input('wait')
            
            y_db += br
            x_db += bc
            y_q, x_q = np.where(lab==label_ignore)
            
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(np.vstack((x_db, y_db)).T)
            positives = knn.kneighbors(np.vstack((x_q, y_q)).T, 1,
                    return_distance=False)
            positives = np.squeeze(positives)
            lab[y_q, x_q] = lab[y_db[positives], x_db[positives]]

    return lab


def extract_connected_components(args, lab0, color_palette=None):
    """
    WARNING: The cc id starts at 1, not 0. 0 is used for the pixels that do not
    belong to any cc (if their blob is too small for example).
    Args:
        lab0: label map. I copy it because I am modifying it.
        min_blob_size: minimum size (in pixels) of semantic blob to be accepted.
    """
    cc = np.zeros(lab0.shape).astype(np.uint8)
    next_cc_id = 1

    lab = lab0.copy()
    lab_id_l = np.unique(lab)
    #print('Label presents: ', lab_id_l)

    for lab_id in lab_id_l:
        if lab_id == 0:
            continue
        # mask all pixels with label lab_id
        l,c = np.where(lab==lab_id)
        #print(l)
        #print(c)
        #print('# lab_id %d: %d'%(lab_id, len(l)))
        mask_lab_id = np.zeros(lab.shape).astype(np.uint8)
        mask_lab_id[l,c] = 255


        while len(l) != 0:
            mask = np.zeros(lab.shape).astype(np.uint8)
            mask[l,c] = 255
            # output of flood fill, 1 where it has filled
            # I don't know why, it does not want to fill with 128
            mask_out = np.zeros((lab.shape[0]+2, lab.shape[1]+2)).astype(np.uint8)
            cv2.floodFill(mask, mask_out, (c[0], l[0]), 128)
            cc_area = np.sum(mask_out[1:-1,1:-1]==1) # blob area
            
            
            # debug: to get a visual idea of a blob size
            #print('cc_id: %d\tsize: %d\treal size: %d'%(next_cc_id-1, len(l), cc_area))
            #cv2.imshow('current cc', (mask_out[1:-1,1:-1]*255).astype(np.uint8))
            #stop_show = cv2.waitKey(0) & 0xFF
            #if stop_show == ord("q"):
            #    exit(0)

            # if this connected components is big enough, keep it
            if cc_area > args.min_blob_size:
                # save new connected component
                cc[mask_out[1:-1, 1:-1]==1] = next_cc_id
                next_cc_id += 1
                
            
            # clear pixels from this connected component
            lab[mask_out[1:-1, 1:-1]==1] = -1
            l, c = np.where(lab==lab_id)

            # color display of flood fill
            if (0==1):
                print('cc_id: %d'%(next_cc_id-1))
                mask_out_col = np.zeros((mask_out.shape) + (3,)).astype(np.uint8)
                mask_out_col[mask_out==1,:] = color_palette[lab_id]
                cv2.imshow('mask_out', mask_out)
                cv2.imshow('mask_out_col', mask_out_col)
                stop_show = cv2.waitKey(0) & 0xFF
                if stop_show == ord("q"):
                    exit(0)

    # color display of all connected components
    if (0==1):
        cc_col = lab2col(cc, color_palette)
        cv2.imshow('cc_col', cc_col)
        stop_show = cv2.waitKey(0) & 0xFF
        if stop_show == ord("q"):
            exit(0)

    return cc


def extract_contours(args, cc):
    """
    Args:
        cc: Connected Components
    """
    contours_l = []
    
    cc_num = np.max(np.unique(cc))

    for j in range(1, cc_num+1): # 0 are all the components to ignore
        cc_j = np.zeros(cc.shape, np.uint8)
        cc_j[cc==j] = 255

        # TODO: delete zigouigoui
        # TODO: delete moving classes
        im2, contours, hierarchy = cv2.findContours(cc_j, cv2.RETR_CCOMP, 
                cv2.CHAIN_APPROX_NONE)
        for k,_ in enumerate(contours):
            contour = contours[k]
            contour_size = contour.shape[0]
            if contour_size > args.min_contour_size:
                contours_l.append(contour)
            #else:
            #    print("This contour is a zigouigoui, ignore it")

    return contours_l


def gimme_gimme_gimme_a_contour(args, sem_img):
    """In my defense, the song just randomly popped up from Youtube ..."""
    # yes, I need this mic mac because I changed the input convention and was
    # too lazy to harmonize everything
    if sem_img.ndim == 3: # it is a color img
        lab = col2lab(sem_img, cst.palette_bgr)
    else:
        lab = sem_img 
    cc = extract_connected_components(args, lab, None)
    contours_l = extract_contours(args, cc)

    return contours_l


def extract_skyline(lab0, min_contour_size, sky_lab_id):
    """
    Args:
        lab0: label map i.e. uint8 np array with values in [0, max_label_id]
        min_contour_size: filter out contour of size < min_contour_size
        sky_lab_id: label if of the 'sky' class
    """
    np.set_printoptions(precision=4)


    rows, cols = lab0.shape[0], lab0.shape[1]

    # get sky pixels
    lab = lab0.copy()
    l,c = np.where(lab==sky_lab_id)
    mask = np.zeros(lab.shape).astype(np.uint8)
    mask[l,c] = 255
    #cv2.imshow('mask_sky', mask)
    #stop_show = cv2.waitKey(0) & 0xFF
    #if stop_show == ord("q"):
    #    exit(0)
    
    
    ## get contours
    im2, contours, hierarchy = cv2.findContours(mask, 
            cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    #all_contour_img = np.zeros(mask.shape, np.uint8)
    #cv2.drawContours(all_contour_img, contours, -1, 255, 1)
    #cv2.imshow('all_contours', all_contour_img)
    #cv2.waitKey(0)

    nbFD = 10 # nb of dft components to keep

    #print(len(contours))
    #print(contours.shape)
    #exit(0)
    
    contours_to_keep = []
    for i,_ in enumerate(contours):
        contour = contours[i]
        contour_size = contour.shape[0]
        if contour_size > min_contour_size:
            contours_to_keep.append(contour)
            continue

            contour_img = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(contour_img, contours, i, 255, 1)
            
            # resample contour to a suitable size for dft
            opt_dft_size = cv2.getOptimalDFTSize(contour_size)
            print('%d: contour_size -> opt_dft_size:\t%d -> %d'%(i, contour_size,
                opt_dft_size))
            sampling = cv2.ximgproc.contourSampling(contour, opt_dft_size)
            print('after sampling, contour_size: %d '% sampling.shape[0])
            sampled_contour_img = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(sampled_contour_img, [sampling.astype(np.int)], 0, 255, 1)
            
            # opencv fourier des of the contour
            des = cv2.ximgproc.fourierDescriptor(sampling, nbElt=opt_dft_size, nbFD = nbFD)
            #des = cv2.ximgproc.fourierDescriptor(contour)
            #print(des.shape)

            # my recoding of fourier des following opencv steps to understand wtf is going on
            #dft = cv2.dft(sampling, flags=cv2.DFT_COMPLEX_OUTPUT)
            #dft = cv2.dft(sampling, flags=cv2.DFT_REAL_OUTPUT)
            #dft = cv2.dft(sampling, flags=cv2.DFT_SCALE)
            dft = cv2.dft(sampling, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
            #print('dft.shape: ', dft.shape)
            n1 = int(nbFD/2)
            n2 = opt_dft_size - n1

            print('nbFD: %d\tn1: %d\tn2: %d'%(nbFD, n1, n2))

            des_me = np.zeros((nbFD,2))
            des_me[:n1,:] = dft[1:n1+1,0,:]
            des_me[n1:,:] = dft[n2:,0,:]

            print('avg contour: ', np.mean(contour, axis=0))
            #print(dft[0,0,:])

            print(np.squeeze(dft)[:10,:])
            print(np.squeeze(des)[:10,:])
            print(np.squeeze(des_me)[:10,:])
            
            ## they are equal
            #print(des)
            #print(des_me)
            
            #print(dft[0:10,:,:])
            
            #idft_contour = cv2.idft(np.squeeze(dft[:400,:,:]), flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
            idft_contour = cv2.idft(np.squeeze(dft), flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
            #print('idft_contour.shape: ', idft_contour.shape)
            
            under_sampled_contour_img = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(under_sampled_contour_img, [idft_contour.astype(np.int)], 0, 255, 1)
           
            cv2.imshow('contour', contour_img)
            cv2.imshow('sampled_contour', sampled_contour_img)
            cv2.imshow('under_sampled_contour', under_sampled_contour_img)
            stop_show = cv2.waitKey(0) & 0xFF
            if stop_show == ord("q"):
                exit(0)


        #    print(des.shape)
        #    des_l.append(np.squeeze(des).flatten())
        #else:
        #    print("%d: ignore. It is a zigouigoui of size: %d"%(i, contour_size))

    return contours_to_keep
        


######################################################

def substract_foreground(lab, min_blob_size=500):
    """
    Delete segmentation of classes to ignore i.e. moving classes such as human,
    cars ...
    """
    
    label_alarm = 0 # set to 1 if patch of foreground is big enough to remove 

    # check if there are foreground labels to ignore
    labels_present = np.unique(lab)
    if np.sum( np.in1d(labels_foreground, labels_present) ) == 0:
        # there is no foreground labels to remove
        lab_filled = lab
    else:
        # there are foreground labels to remove
        # check their size
        labels_foreground_present = labels_foreground[
                np.in1d(labels_foreground, labels_present).nonzero()]
        #print(labels_foreground_present)
        for l in labels_foreground_present:
            l_size = np.sum(lab==l)
            if l_size < min_blob_size:
                continue
            else:
                label_alarm = 1

        if label_alarm == 0: # all foreground regions are too small, ignore them
            lab_filled = lab 
        else:
            lab_masked = lab.copy()
            for label_id in labels_foreground_present:
                lab_masked[lab_masked==label_id] = label_ignore

            lab_filled = lab_masked.copy()
            y_db, x_db = np.where(lab_masked!=label_ignore)
            ##print(y_db.shape)
            #min_y_db, min_x_db = np.min(y_db), np.min(x_db)
            #max_y_db, max_x_db = np.max(y_db), np.max(x_db)
            #lab_rect = lab_filled[min_y_db: max_y_db, min_x_db:max_x_db]
            ##print(lab_rect)
            #y_db, x_db = np.where(lab_rect!=label_ignore)
            ##print(y_db.shape)
            #exit(0)

            #print('Fitting good pixels ...')
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(np.vstack((x_db, y_db)).T)

            for label_id in labels_foreground_present:
                y_q, x_q = np.where(lab==label_id)
                positives = knn.kneighbors(np.vstack((x_q, y_q)).T, 1,
                        return_distance=False)
                positives = np.squeeze(positives)
                lab_filled[y_q, x_q] = lab[y_db[positives], x_db[positives]]
                #for i, (y,x) in enumerate(zip(y_q, x_q)):
                #    lab_filled[y,x] = lab[ y_db[positives[i]], x_db[positives[i]] ]


            #col_masked = lab2col(lab_masked, tools.colors.palette_bgr)

            #cv2.imshow('col', col)
            #cv2.imshow('col_masked', col_masked)
            #stop_show = cv2.waitKey(0) & 0xFF
            #if stop_show == ord("q"):
            #    exit(0)

    #col_filled = lab2col(lab_filled, tools.colors.palette_bgr)
    #cv2.imshow('col_filled', col_filled)
    #cv2.waitKey(0)

    return label_alarm, lab_filled


def dilate(poses_fn):

    meta = np.loadtxt(poses_fn, dtype=str)
    img_fn_l = meta[:,0]
    col_fn_l = sorted(['%s/res/ext_cmu/%s.png'%(cst.SEG_DIR, l.split(".")[0]) for l in img_fn_l])
    img_fn_l = ['%s/%s'%(cst.EXT_IMG_DIR, l) for l in img_fn_l]

    for i, (img_fn, col_fn) in enumerate(zip(img_fn_l, col_fn_l)):

        img = cv2.imread(img_fn)
        col = cv2.imread(col_fn)
        # delete 1-pixel border because the segmentation ouputs 0 on the border
        # and it fucks up everything later ... It took me a day to figure this
        # motherfucker out
        pixel_border = 1
        img = img[pixel_border:-pixel_border, pixel_border:-pixel_border]
        col = col[pixel_border:-pixel_border, pixel_border:-pixel_border]
        col_copy = col.copy()
        
        kernel = np.ones((4,4),np.uint8)
        col = cv2.dilate(col, kernel,iterations = 1)

        cv2.imshow('col_original', col_copy)
        cv2.imshow('col', col)
        stop_show = cv2.waitKey(0) & 0xFF
        if stop_show == ord("q"):
            exit(0)


def extract_semantic_edges(col):
    """
    KO
    """
    k = 3
    blur = cv2.blur(col, (k,k))
    lowThreshold = 50
    ratio = 3
    edges = cv2.Canny(col, lowThreshold, ratio*lowThreshold, 3)

    return edges


def describe_contour_zernike(img_shape, contour, radius):

    contour_img = np.zeros(img_shape[:2], np.uint8)
    cv2.drawContours(contour_img, [contour], 0, 255, 1)
    #contour = np.squeeze(contour)
    #print(contour.shape)

    des = mahotas.features.zernike_moments(contour_img, radius)
    #print(des.shape)

    return des


def test_substract_foreground(slice_id, cam_id, survey_id, write=False):

    #slice_id = 23
    #cam_id = 0
    #survey_id = 6

    if survey_id == -1:
        poses_fn = '%s/surveys/%d/pose/c%d_db.txt'%(cst.META_DIR, slice_id, cam_id)
    else:
        poses_fn = '%s/surveys/%d/pose/c%d_%d.txt'%(cst.META_DIR, slice_id, cam_id, survey_id)
    
    meta = np.loadtxt(poses_fn, dtype=str)
    img_fn_l = meta[:,0]
    col_fn_l = sorted(['%s/%s.png'%(cst.SEG_DIR, l.split(".")[0]) for l in
            img_fn_l])
    img_fn_l = ['%s/%s'%(cst.EXT_IMG_DIR, l) for l in img_fn_l]
    
    # output images
    col_clean_fn_l = [l.replace('ext_cmu', 'ext_cmu_clean') for l in col_fn_l]
    col_clean_dirs = set([os.path.dirname(l) for l in col_clean_fn_l])
    for dirname in col_clean_dirs: # create output dirs
        if not os.path.exists(dirname):
            os.makedirs(dirname)


    for i, col_fn in enumerate(col_fn_l):
        #print('col_fn: %s'%col_fn)
        col = cv2.imread(col_fn)

        # delete 1-pixel border because the segmentation ouputs 0 on the border
        # and it fucks up everything later ... It took me a day to figure this
        # motherfucker out
        pixel_border = 1
        col = col[pixel_border:-pixel_border, pixel_border:-pixel_border]
        col = cv2.resize(col, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        lab = col2lab(col, cst.palette_bgr)
        label_alarm, lab_clean = substract_foreground(lab)
        
        if label_alarm == 1:
            print('KO: %d/%d'%(i, len(col_fn_l)))
            if write:
                cv2.imwrite(col_clean_fn_l[i], lab_clean)
            else:
                img_fn = img_fn_l[i]
                img = cv2.imread(img_fn)
                img = img[pixel_border:-pixel_border, pixel_border:-pixel_border]
                img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

                #print('colorize lab_clean ...')
                col_clean = lab2col(lab_clean, cst.palette_bgr)
                #print('... OK')
                cv2.imshow(' img | col | col_clean', np.hstack( (img, col,
                    col_clean)) )
                stop_show = cv2.waitKey(0) & 0xFF
                if stop_show == ord("q"):
                    exit(0)
        else:
            print('OK: %d/%d'%(i, len(col_fn_l)))


def show_substract_foreground():


    slice_id = 24
    cam_id = 0
    survey1_id = 1
    poses_fn = '%s/surveys/%d/pose/c%d_db.txt'%(cst.META_DIR, slice_id, cam_id)
    
    meta = np.loadtxt(poses_fn, dtype=str)
    img_fn_l = meta[:,0]
    col_fn_l = sorted(['%s/res/ext_cmu/%s.png'%(cst.SEG_DIR, l.split(".")[0]) for l in img_fn_l])
    # output images
    lab_clean_fn_l = [l.replace('ext_cmu', 'ext_cmu_clean') for l in col_fn_l]

    for i, (col_fn, lab_clean_fn) in enumerate(zip(col_fn_l, lab_clean_fn_l)):
        if not os.path.exists(lab_clean_fn):
            print('Clean %d/%d'%(i, len(col_fn_l)))
            continue
        else:
            print('New   %d/%d'%(i, len(col_fn_l)))

        col = cv2.imread(col_fn)
        # delete 1-pixel border because the segmentation ouputs 0 on the border
        # and it fucks up everything later ... It took me a day to figure this
        # motherfucker out
        pixel_border = 1
        #img = img[pixel_border:-pixel_border, pixel_border:-pixel_border]
        col = col[pixel_border:-pixel_border, pixel_border:-pixel_border]

        
        lab_clean = cv2.imread(lab_clean_fn, cv2.IMREAD_UNCHANGED)
        col_clean = lab2col(lab_clean, cst.palette_bgr)

        cv2.imshow('col | col_clean', np.hstack((col, col_clean)))
        stop_show = cv2.waitKey(0) & 0xFF
        if stop_show == ord("q"):
            exit(0)





if __name__=='__main__':
    
    slice_id = 24
    cam_id = 0
    survey1_id = 1

    test()

    #write = (1==1)
    #cam_id = 0
    #for slice_id in range(24, 26):
    #    for survey_id in range(-1, 11):
    #        test_substract_foreground(slice_id, cam_id, survey_id, write)

    #show_substract_foreground()

    #poses_fn = '%s/surveys/%d/pose/c%d_db.txt'%(cst.META_DIR, slice_id, cam_id)
    #dilate(poses_fn)


