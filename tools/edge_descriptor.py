
import numpy as np
import cv2
import pywt

#def describe_one_img(args, img_fn):
#
#    min_blob_size = 1000
#    min_contour_size = 200
#
#    des_l = []
#    
#    img = cv2.imread(img_fn)
#    # cut border because of artefacts
#    img = img[args.pixel_border:-args.pixel_border, args.pixel_border:-args.pixel_border]
#
#    lab = tools_sem.col2lab(img, color_tools.palette_bgr)
#    cc = tools_sem.extract_connected_components(lab, min_blob_size,
#            None)
#
#    cc_num = np.max(np.unique(cc)) + 1
#    for j in range(1, cc_num): # 0 are ignore components
#        cc_j = np.zeros(cc.shape, np.uint8)
#        cc_j[cc==j] = 255
#
#        # TODO: delete zigouigoui
#
#        # TODO: delete moving classes
#        
#        if args.method == 'zernike':
#            des = mahotas.features.zernike_moments(cc_j, zernike_radius)
#        elif args.method == 'hu':
#            des = cv2.HuMoments(cv2.moments(cc_j)).flatten()
#        elif args.method == 'fourier':
#            im2, contours, hierarchy = cv2.findContours(cc_j, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
#            for k,_ in enumerate(contours):
#                contour = contours[k]
#                contour_size = contour.shape[0]
#                if contour_size > min_contour_size:
#                    sampling = cv2.ximgproc.contourSampling(contour, 100)
#                    des = cv2.ximgproc.fourierDescriptor(sampling)
#                #else:
#                #    print("This contour is a zigouigoui, ignore it")
#        else:
#            print("Error: wtf is this descriptor: %s"%args.des)
#            exit(1)
#        des_l.append(des)
#
#    return np.vstack(des_l)


def describe_fourier(cc, min_contour_size):
    """
    Connected components
    """
    
    des_l = []
    cc_num = np.max(np.unique(cc))
    for i in range(1, cc_num): # 0 are the ignored components
            cc_i = np.zeros(cc.shape, np.uint8)
            cc_i[cc==i] = 255
            #cv2.imshow('cc_i', cc_i)
            #stop_show = cv2.waitKey(0) & 0xFF
            #if stop_show == ord("q"):
            #    exit(0)
            #continue
            
            # contours
            if (1==1):
                im2, contours, hierarchy = cv2.findContours(cc_i, 
                        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

                #contours_img = np.zeros(img.shape, np.uint8)
                #cv2.drawContours(contours_img, contours, -1, (0,255,0), 3)
                for i,_ in enumerate(contours):
                    contour = contours[i]
                    
                    contour_img = np.zeros(cc.shape, np.uint8)
                    cv2.drawContours(contour_img, [contour], 0, 255, 1)

                    contour_size = contour.shape[0]
                    if contour_size > min_contour_size:
                        #print('%d: contour_size: %d'%(i, contour_size))
                        sampling = cv2.ximgproc.contourSampling(contour, 100)
                        des = cv2.ximgproc.fourierDescriptor(sampling)
                        #print(des.shape)
                        des_l.append(np.squeeze(des).flatten())
                    #else:
                    #    print("%d: ignore. It is a zigouigoui of size: %d"%(i, contour_size))
                               
                    #cv2.imshow('contour', contour_img)
                    #stop_show = cv2.waitKey(0) & 0xFF
                    #if stop_show == ord("q"):
                    #    exit(0)
    
    return des_l


def angle_tangeant(z0, z1):
    """
    Computes the oriented angle between z0 and z1 where the positive sens is
    the trigonometric sens.
    Args:
        z0: (x0, y0)
        z1: (x1, y1)
    """
    #print('z0 -> z1: %d, %d -> %d, %d'%(z0[0], z0[1], z1[0], z1[1]))
    if ( z1[0] - z0[0]) == 0: # vertical line
        if (z1[1] < z0[1]): # going up
            a = np.pi/2
        elif (z1[1] > z0[1]): # going down
            a = 3 * np.pi/2
        else:
            print('Error: you are measuring an angle between two equal points')
            print(z0)
            print(z1)
            exit(1)
    else: # slope != infinity so we can measure it
        a = np.arctan2( float(z1[1] - z0[1]) , float(z1[0] - z0[0]) )
    return a


#def contour2tangeant(contour, sample_num, contour_img=None):
#    """
#    Compute curve representation with the cumulative angle on N points of the
#    curve. The curve is normalised to have new length l=2pi.
#    Args:
#        contour: (N, 1, 2)
#        sample_num: 
#        contour_img: img of the contour, for debug
#    """
#
#    # normalise curve parameter 
#    L = cv2.arcLength(contour, True)
#    contour_N = contour.shape[0] # num of pts
#    step = int((contour_N-1) / sample_num)
#    s = 0
#    t = lambda x: 2 * np.pi * x / L # reparametrisation
#    
#    contour = np.squeeze(contour)
#    z0 = (contour[0,0], contour[0,1]) # Z(0) (x_o, y_o)
#
#
#    # compute \delta_0 = \theta(0)
#    z1 = (contour[step, 0], contour[step, 1])
#    cv2.circle(contour_img, z1, 2, 255, -1, lineType=16)
#    d0 = angle_tangeant(z0, z1)
#    if contour_img is not None:
#        cv2.circle(contour_img, z0, 2, 255, -1, lineType=16)
#        cv2.line(contour_img, z0, z1, 200, 1, 8, 0)
#    
#    #print("\n **Sample phi(t) ** ")
#    phi_l = []
#    phi_l.append(d0)
#    for j in range(2, sample_num+1):
#        idx = int(step*j)
#        print('j: %d\tidx: %d'%(j, idx))
#        #zj = tuple(contour[idx,:])
#        s = cv2.arcLength(contour[:idx, :], False)
#        dj = angle_tangeant(contour[idx-1,:], contour[idx,:])
#        phi_l.append( dj - t(s) )
#
#        if contour_img is not None:
#            cv2.circle(contour_img, tuple(contour[idx,:]), 2, 255, -1, lineType=16)
#
#    return np.array(phi_l)

def contour2tangeant(contour, sample_num, contour_img=None):
    """
    Compute curve representation with the cumulative angle on N points of the
    curve. The curve is normalised to have new length l=2pi.
    Args:
        contour: (N, 1, 2)
        sample_num: 
        contour_img: img of the contour, for debug
    """

    # normalise curve parameter 
    L = cv2.arcLength(contour, True)
    contour_N = contour.shape[0] # num of pts
    step = int((contour_N-1) / sample_num)
    s = 0
    t = lambda x: 2 * np.pi * x / L # reparametrisation
    
    if contour.ndim == 2:
        contour = np.expand_dims(contour, 1)

    #print(contour.shape) 
    contour = cv2.ximgproc.contourSampling(contour, sample_num+1)
    #print(contour.shape)
    #exit(0)
    
    contour = np.squeeze(contour)
    z0 = (contour[0,0], contour[0,1]) # Z(0) (x_o, y_o)


    # compute \delta_0 = \theta(0)
    z1 = (contour[1, 0], contour[1, 1])
    cv2.circle(contour_img, z1, 2, 255, -1, lineType=16)
    d0 = angle_tangeant(z0, z1)
    if contour_img is not None:
        cv2.circle(contour_img, z0, 2, 255, -1, lineType=16)
        cv2.line(contour_img, z0, z1, 200, 1, 8, 0)
    
    #print("\n **Sample phi(t) ** ")
    phi_l = []
    phi_l.append(d0)
    for idx in range(2, sample_num+1):
        #idx = int(step*j)
        #zj = tuple(contour[idx,:])
        s = cv2.arcLength(contour[:idx, :], False)
        #print('j: %d\tl: %.2f'%(idx, s))
        dj = angle_tangeant(contour[idx-1,:], contour[idx,:])
        phi_l.append( dj - t(s) )

        if contour_img is not None:
            cv2.circle(contour_img, tuple(contour[idx,:]), 2, 255, -1, lineType=16)
    #print('len(phi_l): %d'%len(phi_l))

    return np.array(phi_l)


def fourier_zahn72(contours, img_shape, nbFD=-1):

    if len(contours)==0:
        print("Error: list of contours is empty.")
        return None

    des_l = []
    
    # find longest contour
    contour_size = -1
    contour_idx = -1
    for i,contour in enumerate(contours):
        if contour.shape[0] > contour_size:
            contour_size = contour.shape[0]
            contour_idx = i
    #print(contour_idx)
    contour = contours[contour_idx]
    #print(contour)
    #print(contour.shape)


    # draw contour
    contour_img = np.zeros(img_shape, np.uint8)
    cv2.drawContours(contour_img, [contour], 0, 100, 1)
    
    sample_num = 128 # TODO: un-hard this 
    data = contour2tangeant(phi_l, sample_num, contour_img)
    data_size = data.shape[0]
    print('data_size: %d'%data_size)
    
    dft = cv2.dft(data, flags=cv2.DFT_REAL_OUTPUT)
    #dft = cv2.dft(data, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
    #dft = cv2.dft(data, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft = np.squeeze(dft)
    #print('dft.shape: %d', dft.shape[0])
    
    # keep only nbFD components
    if nbFD != -1:
        n1 = int(nbFD/2)
        n2 = data_size - n1
        des_me = np.zeros(nbFD)
        #print('nbFD: %d\tn1: %d\tn2: %d'%(nbFD, n1, n2))
        #print(des_me[:n1].shape)
        #print(dft[1:n1+1].shape)
        #print(des_me[n1:].shape)
        #print(dft[n2:].shape)
        
        des_me[:n1] = dft[1:n1+1]
        des_me[n1:] = dft[n2:]
    else:
        des_me = dft
    
    # test that the 0 component is the DC signal
    #print('avg data: %.3f\tc_0: %.3f' %(np.mean(data, axis=0), dft[0]))

    #print('dft')
    #print(np.squeeze(dft)[:10])
    #print(np.squeeze(des_me)[:10])
    #cv2.imshow('contour', contour_img)
    #stop_show = cv2.waitKey(0) & 0xFF
    #if stop_show == ord("q"):
    #    exit(0)

    return des_me


def fourier_granlund72(contours, img_shape, nbFD=-1):
    """
    Complex dft on N points sampled from the curve.
    Same as cv::fourierDescriptor. The only possible change is on to sample on
    the curve.
    """

    if len(contours)==0:
        print("Error: list of contours is empty.")
        return None

    # find longest contour
    contour_size = -1
    contour_idx = -1
    for i,contour in enumerate(contours):
        if contour.shape[0] > contour_size:
            contour_size = contour.shape[0]
            contour_idx = i
    contour = contours[contour_idx]
    #print('contour.shape: ', contour.shape)

    # draw contour
    if (0==1):
        contour_img = np.zeros(img_shape, np.uint8)
        cv2.drawContours(contour_img, [contour], 0, 100, 1)
    
    sample_num = 128
    if (0==1): # my code
        # sample points TODO: optimise
        contour_N = contour.shape[0] # num of pts
        step = int((contour_N-1) / sample_num)
        pt_l = []
        for j in range(sample_num):
            idx = int(step*j)
            pt_l.append(contour[idx, :])
        data = np.array(pt_l).astype(np.float32)
        data_size = data.shape[0]
        #print('data.shape: ', data.shape)
        #print('data_size: %d'%data_size)
    
        dft = cv2.dft(data, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
        dft = np.squeeze(dft)
        #print('dft.shape: ', dft.shape)
 
        # keep only nbFD fourier components
        if nbFD != -1:
            n1 = int(nbFD/2)
            n2 = data_size - n1
            des = np.zeros((nbFD,2))
            des[:n1,:] = dft[1:n1+1,:]
            des[n1:,:] = dft[n2:,:]

    else: # opencv code
    
        sampling = cv2.ximgproc.contourSampling(contour, sample_num)
        #print('sampling.shape: ', sampling.shape)
        des = cv2.ximgproc.fourierDescriptor(sampling, nbFD=nbFD)
    
    # transform it into 1D vector
    # use only magnitude for now
    des = np.squeeze(des)
    des = np.sqrt(np.sum( des**2, axis=1) )

    return des


def wavelet_chuang96(args, contour, img_shape):


    # draw contour
    if (0==1):
        contour_img = np.zeros(img_shape, np.uint8)
        cv2.drawContours(contour_img, [contour], 0, 100, 1)
    else:
        contour_img = None
    
    # TODO: replace with more relevant sampling than just regular steps
    #sample_num = 32 # must be a power of 2
    base_sample_num = np.log(args.contour_sample_num)/np.log(2)
    if 2**base_sample_num != args.contour_sample_num:
        print("Error: you must sample 'a power of 2' number of points.")
        exit(1)
    
    wt_name = 'haar'
    wt = pywt.Wavelet(wt_name)
    #print(w)

    # 1d on the tangeant representation
    if (0==1): 
        #sample_num = 64 # TODO: un-hard this 
        data = contour2tangeant(contour, args.contour_sample_num, contour_img)

        input_size = data.shape[0]
        #print('input_size: %d'%input_size)
        #return None
        #print('# of filters: %d'%w.dec_len)
        output_size = pywt.dwt_coeff_len(input_size, filter_len= wt.dec_len,
                mode='symmetric')
        #print('output_size: %d'%output_size)

        cA, cD = pywt.dwt(data, wt) # = wavedec(..., level=1)
        #print('\ncA')
        #print(cA)
        #print(cA)
        #print(cA.shape)
        #print('\ncD')
        #print(cD)
        #print(cD.shape)
        toto = pywt.wavedec(data, wt, level=1) # decomposition in 
        cA, cD = toto
        #print(toto)
        des = pywt.coeffs_to_array(toto)[0]
        #print(des)

        #print('\des.shape')
        #print(des.shape)

    else:
        #print(contour.shape)
        if contour.ndim == 2:
            contour = np.expand_dims(contour, 1)
        sampling = cv2.ximgproc.contourSampling(contour, args.contour_sample_num)
        sampling = np.squeeze(sampling)

        #x_cA, x_cD = pywt.wavedec(sampling[:,0], wt, level=1) # = wavedec(..., level=1)
        #y_cA, y_cD = pywt.wavedec(sampling[:,1], wt, level=1) # = wavedec(..., level=1)

        xt = pywt.wavedec(sampling[:,0], wt, level=1) # = wavedec(..., level=1)
        yt = pywt.wavedec(sampling[:,1], wt, level=1) # = wavedec(..., level=1)

        x_des = pywt.coeffs_to_array(xt)[0]
        y_des = pywt.coeffs_to_array(yt)[0]
        
        des = np.hstack((x_des, y_des))

        des /= np.sqrt( np.sum(des**2) )
        #print(des.shape)

        #toto = pywt.wavedec2(sampling, 'db2')
        #print(len(toto))
        #print(toto)
    
    return des

