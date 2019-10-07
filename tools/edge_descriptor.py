"""Implementation of various edge descriptors."""
import numpy as np
import cv2
import pywt


def describe_semantic_edge(params, curves_d):
    des_d = {}
    for label_id, curves_l in curves_d.items():
        des_d[label_id] = []
        for c in curves_l:
            des_d[label_id].append( wavelet_chuang96(params, c, None))
    return des_d


def fourier_opencv(cc, min_contour_size):
    """Edge description using OpenCV implementation of fourier descriptor."""
    des_l = []
    cc_num = np.max(np.unique(cc))
    for i in range(1, cc_num): # 0 are the ignored components
            cc_i = np.zeros(cc.shape, np.uint8)
            cc_i[cc==i] = 255
            
            # contours
            if (1==1):
                im2, contours, hierarchy = cv2.findContours(cc_i, 
                        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

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
    """Implements the fourier descriptor from 
    C. T. Zahn and R. Z. Roskies, “Fourier descriptors for plane closed
    curves,” IEEE Transactions on computers, vol. 100, no. 3, pp. 269–
    281, 1972.
    """
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

    # draw contour
    contour_img = np.zeros(img_shape, np.uint8)
    cv2.drawContours(contour_img, [contour], 0, 100, 1)
    
    sample_num = 128 # TODO: un-hard this 
    data = contour2tangeant(phi_l, sample_num, contour_img)
    data_size = data.shape[0]
    print('data_size: %d'%data_size)
    
    dft = cv2.dft(data, flags=cv2.DFT_REAL_OUTPUT)
    dft = np.squeeze(dft)
    
    # keep only nbFD components
    if nbFD != -1:
        n1 = int(nbFD/2)
        n2 = data_size - n1
        des_me = np.zeros(nbFD)
        des_me[:n1] = dft[1:n1+1]
        des_me[n1:] = dft[n2:]
    else:
        des_me = dft
    return des_me


def fourier_granlund72(contours, img_shape, nbFD=-1):
    """Implements the fourier descriptor from
    C. T. Zahn and R. Z. Roskies, “Fourier descriptors for plane closed
    curves,” IEEE Transactions on computers, vol. 100, no. 3, pp. 269–
    281, 1972.
    
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
    """Implements the wavelet descriptor from 
    G.-H. Chuang and C.-C. Kuo, “Wavelet descriptor of planar curves:
    Theory and applications,” IEEE Transactions on Image Processing,
    vol. 5, no. 1, pp. 56–70, 1996
    Args:
        args: misc parameters
        contour: list of contours. Contour shape is [N,1,2] with N the # of pts
            in the contour.
        img_shape: (h,w)
    """
    # draw contour
    if (0==1):
        contour_img = np.zeros(img_shape, np.uint8)
        cv2.drawContours(contour_img, [contour], 0, 100, 1)
    else:
        contour_img = None
    
    # TODO: replace with more relevant sampling than just regular steps
    base_sample_num = np.log(args.contour_sample_num)/np.log(2)
    if 2**base_sample_num != args.contour_sample_num:
        print("Error: you must sample 'a power of 2' number of points.")
        exit(1)
    
    wt_name = 'haar'
    wt = pywt.Wavelet(wt_name)

    if contour.ndim == 2:
        contour = np.expand_dims(contour, 1)
    sampling = cv2.ximgproc.contourSampling(contour, args.contour_sample_num)
    sampling = np.squeeze(sampling)

    xt = pywt.wavedec(sampling[:,0], wt, level=1) # = wavedec(..., level=1)
    yt = pywt.wavedec(sampling[:,1], wt, level=1) # = wavedec(..., level=1)

    x_des = pywt.coeffs_to_array(xt)[0]
    y_des = pywt.coeffs_to_array(yt)[0]
    
    des = np.hstack((x_des, y_des))
    des /= np.sqrt( np.sum(des**2) )
    return des

