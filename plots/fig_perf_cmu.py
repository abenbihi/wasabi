"""Perf plots."""
import os

import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

###############################################################################
# PAPER PLOTS

RANDOM_DIR  = '../fuck_retrieval/res/random/'
FUCK_DIR    = '../fuck_retrieval/res/fuck/'
NETVLAD_DIR = '../fuck_netvlad/res/'

metric_l = ['mAP', 'rec1', 'rec5', 'rec10', 'rec20']
n_values = [1, 5, 10, 20]


method_l = [
        'random', 'wasabi', 
        'netvlad', 'netvlad*', 
        'delf', 
        'vlad', 'vlad*',
        'bow', 'bow*',
        ]

color_l     = [
        'gray', # random
        'green', # wasabi
        'blue', 'red', # netvlad, netvlad*
        'orange',  # delf
        'hotpink', 'midnightblue', # vlad, vlad*
        'purple', 'crimson', # bow, bow*
        ]

res_dir_l   = [
        '../fuck_retrieval/res/random/', 
        '../fuck_retrieval/res/fuck/', 
        '../fuck_netvlad/res/', '../fuck_netvlad/res/', 
        '../fuck_retrieval/res/vlad/', # delf
        '../fuck_retrieval/res/vlad/', '../fuck_retrieval/res/vlad', # vlad, vlad*
        '../fuck_retrieval/res/vlad/', '../fuck_retrieval/res/vlad/', # bow, bow*
        ]

fmt_l = ['.', 'o', 
        '^', '^', 
        'x', 
        'v', 'v',
        '+', '+']

capsize_l = [1, 5, 2, 3, '3', 4, 4]


def perf_rec_avg_surveys(perf, slice_v, cam_v, plot_dir, plot_fn):
    """Average the recalls over surveys for each slice. Shows the perf vs the
    semantic elements."""
    survey_num = perf['random']['mAP'].shape[0]
    slice_num = slice_v.shape[0]
    survey_ok = np.ones(survey_num, np.int32)
    survey_ok[8] = 0

    #method_ok_l = list(perf.keys())

    # avg over survey for each slice
    methods_num = len(method_l)
    for metric in metric_l[1:]:
        for method in method_l:
            if perf[method]['trial'] == -1:
                continue
            print('method: %s'%method)
            avg = np.zeros(slice_num)
            std = np.zeros(slice_num)
            for i, (slice_id, cam_id) in enumerate(zip(slice_v, cam_v)):
                print('slice_id: %d\tcam_id: %d'%(slice_id, cam_id))
                if (slice_id in [24,25]) and (cam_id == 1):
                    avg[i] = np.mean(perf[method][metric][survey_ok==1,i])
                    std[i] = np.std(perf[method][metric][survey_ok==1,i])
                else:
                    avg[i] = np.mean(perf[method][metric][:,i])
                    std[i] = np.std(perf[method][metric][:,i])
             
            perf[method]['avg_survey_%s'%metric] = avg
            perf[method]['std_survey_%s'%metric] = std

    # plot recall@N for each slice_cam
    for j, (slice_id, cam_id) in enumerate(zip(slice_v, cam_v)):
        plt.figure()
        for i, method in enumerate(method_l):
            if perf[method]['trial'] == -1:
                continue
            print('method: %s'%method)
            color = color_l[i]
            
            avg_rec_l = []
            std_rec_l = []
            for metric in metric_l[1:]:
                avg_rec_l.append(perf[method]['avg_survey_%s'%metric][j])
                std_rec_l.append(perf[method]['std_survey_%s'%metric][j])
            avg_rec_v = np.array(avg_rec_l)
            std_rec_v = np.array(std_rec_l)
            plt.plot(n_values, avg_rec_v, label=method, color=color, marker=fmt_l[i])
            if method in ['wasabi', 'netvlad']: # for better display
                alpha = 0.1
            else:
                alpha = 0.03
            plt.fill_between(n_values, avg_rec_v + std_rec_v, avg_rec_v -
                    std_rec_v, facecolor=color, alpha=alpha)
        
        fig_fn = '%s/recN_%d_c%d.png'%(plot_dir, slice_id, cam_id)
        plt.xlabel('N')
        plt.ylabel('Recall@N')
        plt.axis([0, 21, -.1, 1.1])
        plt.xticks(n_values)
        if j == 0:
            plt.legend(loc=0)
        plt.title('Slice %d - Camera %d'%(slice_id, cam_id))
        plt.savefig(fig_fn)
        plt.close()
    
    # assemble the plots
    plot_num = slice_v.shape[0]
    img_num_per_col = 3
    line_num = int(np.ceil(plot_num/img_num_per_col))
    
    line_d = {}
    for line in range(line_num):
        line_d[line] = []
    
    for j, (slice_id, cam_id) in enumerate(zip(slice_v, cam_v)):
        fig_fn = "%s/recN_%d_c%d.png"%(plot_dir, slice_id, cam_id)
        fig = cv2.imread(fig_fn)
        line = j//3
        print(j, j//3)
        line_d[line].append(fig)

    # complete line2
    col_fill_num = line_num * img_num_per_col - plot_num
    fig = cv2.imread("%s/recN_%d_c%d.png"%(plot_dir, slice_id, cam_id))
    print("line_num-1: %d"%(line_num-1))
    for _ in range(col_fill_num):
        line_d[line_num-1].append(255*np.ones(fig.shape, np.uint8))
    
    # stack the plot lines
    img_line_d = {}
    for i in range(line_num):
        img_line_d[i] = np.hstack(line_d[i])
        cv2.imshow("img_line_d[i]", img_line_d[i])
        cv2.waitKey(0)
    # careful, it can stack them in non-sorted order
    out = np.vstack(list(img_line_d.values())) 
    cv2.imwrite(plot_fn, out)
    cv2.imshow('out', out)
    cv2.waitKey(0)


###############################################################################
def perf_rec_avg_slices(perf, survey_ok, slice_v, cam_v, plot_dir, plot_fn):
    """Average the recalls over surveys for each slice. Shows the perf vs the
    semantic elements. """

    survey_num = 9 #perf['random']['mAP'].shape[0]
    slice_num = slice_v.shape[0]
    
    # avg over survey for each slice
    methods_num = len(method_l)
    for metric in metric_l[1:]:
        for method in method_l:
            if perf[method]['trial'] == -1:
                continue
            print('method: %s'%method)
            avg = np.zeros(survey_num)
            std = np.zeros(survey_num)
            for survey_id in range(survey_num):
                #print('survey_id: %d'%survey_id)
                if survey_id == 8:
                    keep = np.where(survey_ok==1)[0]
                    avg[survey_id] = np.mean(perf[method][metric][survey_id,keep])
                    std[survey_id] = np.std(perf[method][metric][survey_id,keep])
                else:
                    avg[survey_id] = np.mean(perf[method][metric][survey_id, :])
                    std[survey_id] = np.std(perf[method][metric][survey_id, :])
             
            perf[method]['avg_survey_%s'%metric] = avg
            perf[method]['std_survey_%s'%metric] = std

    
    # plot recall@N for each slice_cam
    for survey_id in range(survey_num):
        plt.figure()
        for i, method in enumerate(method_l):
            if perf[method]['trial'] == -1:
                continue
            print('method: %s'%method)
            color = color_l[i]
            avg_rec_l = []
            std_rec_l = []
            for metric in metric_l[1:]:
                avg_rec_l.append(perf[method]['avg_survey_%s'%metric][survey_id])
                std_rec_l.append(perf[method]['std_survey_%s'%metric][survey_id])
            avg_rec_v = np.array(avg_rec_l)
            std_rec_v = np.array(std_rec_l)
            plt.plot(n_values, avg_rec_v, label=method, color=color, marker=fmt_l[i])
            if method in ['wasabi', 'netvlad']:
                alpha = 0.1
            else:
                alpha = 0.03
            plt.fill_between(n_values, avg_rec_v + std_rec_v, avg_rec_v -
                    std_rec_v, facecolor=color, alpha=alpha)
        
        fig_fn = "%s/recN_survey_%d.png"%(plot_dir, survey_id)
        plt.xlabel('N')
        plt.ylabel('Recall@N')
        plt.axis([0, 21, -.1, 1.1])
        plt.xticks(n_values)
        if survey_id == 0:
            plt.legend(loc=0)
        plt.title('Traversal %d'%survey_id)
        plt.savefig(fig_fn)
        plt.close()


    # assmble plots
    img_num_per_col = 3
    plot_num = survey_num
    line_num = int(np.ceil(plot_num/img_num_per_col))
    
    line_d = {}
    for line in range(line_num):
        line_d[line] = []
    
    for j in range(survey_num):
        fig_fn = "%s/recN_survey_%d.png"%(plot_dir, j)
        fig = cv2.imread(fig_fn)
        line = j//img_num_per_col
        print(j, j//img_num_per_col)
        line_d[line].append(fig)

    # complete line2
    col_fill_num = line_num * img_num_per_col - plot_num
    fig = cv2.imread("%s/recN_survey_0.png"%(plot_dir))
    print("line_num-1: %d"%(line_num-1))
    for _ in range(col_fill_num):
        line_d[line_num-1].append(255*np.ones(fig.shape, np.uint8))
    
    # stack the plot lines
    img_line_d = {}
    for i in range(line_num):
        img_line_d[i] = np.hstack(line_d[i])
        cv2.imshow("img_line_d[i]", img_line_d[i])
        cv2.waitKey(0)
    # careful, it can stack them in non-sorted order
    out = np.vstack(list(img_line_d.values())) 
    cv2.imwrite(plot_fn, out)
    cv2.imshow('out', out)
    cv2.waitKey(0)

    #line_d[2].append(255*np.ones(fig.shape, np.uint8))
    #line_d[2].append(255*np.ones(fig.shape, np.uint8))
    #print(len(line_d[0]))
    #print(len(line_d[1]))
    #out = np.vstack((np.hstack(line_d[0]), np.hstack(line_d[1]),
    #    np.hstack(line_d[2])))
    #cv2.imwrite('plots/img/rec_avg_over_slice.png', out)
    #cv2.imshow('out', out)
    #cv2.waitKey(0)


def fig5():
    """Perf plot on cmu park."""
    slice_v = np.array([22, 23, 24, 25])
    cam_v = np.array([0, 1])
    slice_num = slice_v.shape[0]
    slice_v = np.tile(slice_v.reshape(slice_num,1), (1,2)).flatten()
    cam_v = np.tile(cam_v, (slice_num))
    
    trial_l     = [
            0, # random
            7, # wasabi
            7, 9, # netvlad, netvlad*
            34, # delf
            23, 28, # vlad, vlad*
            25, 27, # bow, bow*
            ]

    perf = {}
    survey_num = 10 # TODO: delete once you delete survey 10
    for i, method in enumerate(method_l):
        perf[method] = {}
        perf[method]['trial'] = trial_l[i]
        perf[method]['res_dir'] = res_dir_l[i]
        for metric in metric_l:
            perf[method][metric] = np.loadtxt('%s/%d/%s.txt'%(res_dir_l[i],
                trial_l[i], metric))[:survey_num, :]

    #plot_dir = "plots/img/fig5"
    #if not os.path.exists(plot_dir):
    #    os.makedirs(plot_dir)
    #perf_rec_avg_surveys(perf, slice_v, cam_v, plot_dir, "%s.png"%plot_dir)


    # take into account the lack of ground-truth poses in surveys 24/25 cam1
    # survey 8
    survey_ok = np.ones(slice_v.shape[0], np.int32)
    survey_ok[5] = 0
    survey_ok[7] = 0

    plot_dir = "plots/img/fig6"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    perf_rec_avg_slices(perf, survey_ok, slice_v, cam_v, plot_dir, "%s.png"%plot_dir)



def fig9():
    """Perf plot on cmu urban."""
    slice_v = np.array([6, 7, 8])
    cam_v = np.array([0, 1])
    slice_num = slice_v.shape[0]
    slice_v = np.tile(slice_v.reshape(slice_num,1), (1,2)).flatten()
    cam_v = np.tile(cam_v, (slice_num))
    
    trial_l = [
            3, # random
            19, # wasabi
            11, -1, # netvlad
            35, # delf
            19, -1,# vlad
            26, -1, # bow
            ]

    perf = {}
    survey_num = 10 # TODO: delete once you delete survey 10
    for i, method in enumerate(method_l):
        perf[method] = {}
        perf[method]['trial'] = trial_l[i]
        if trial_l[i] == -1:
            continue
        perf[method]['res_dir'] = res_dir_l[i]
        for metric in metric_l:
            perf[method][metric] = np.loadtxt('%s/%d/%s.txt'%(res_dir_l[i],
                trial_l[i], metric))[:survey_num, :]

    #plot_dir = "plots/img/fig9"
    #if not os.path.exists(plot_dir):
    #    os.makedirs(plot_dir)
    #perf_rec_avg_surveys(perf, slice_v, cam_v, plot_dir, "%s.png"%plot_dir)

    # here, all surveys have ground-truth camera poses
    survey_ok = np.ones(slice_v.shape[0], np.int32)
    plot_dir = "plots/img/fig10"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    perf_rec_avg_slices(perf, survey_ok, slice_v, cam_v, plot_dir, "%s.png"%plot_dir)


if __name__=='__main__':
    #fig5()
    fig9()


    #perf_rec_avg_surveys()
    #perf_rec_avg_slices()
    
    #perf_mAP_avg_surveys()
    #perf_mAP_avg_slices()



# TODO
#def perf_mAP_avg_surveys():
#    """Average the mAP over surveys for each slice. Shows the perf vs the
#    semantic elements. """ 
#    # avg over survey for each slice
#    metric = 'mAP'
#    
#    survey_ok = np.ones(11, np.int32)
#    survey_ok[8] = 0
#
#    methods_num = len(method_l)
#    for method in method_l:
#        print('method: %s'%method)
#        avg = np.zeros(slice_num)
#        std = np.zeros(slice_num)
#        for i, slice_cam in enumerate(slice_cam_id):
#            slice_id = int(slice_cam/10)
#            cam_id = slice_cam%10
#            print('slice_id: %d\tcam_id: %d'%(slice_id, cam_id))
#            if (slice_id in [24,25]) and (cam_id == 1):
#                #print('STOPPP')
#                #print(perf[method][metric][:,i])
#                #print(perf[method][metric][survey_ok==1,i])
#                avg[i] = np.mean(perf[method][metric][survey_ok==1,i])
#                std[i] = np.std(perf[method][metric][survey_ok==1,i])
#            else:
#                avg[i] = np.mean(perf[method][metric][:,i])
#                std[i] = np.std(perf[method][metric][:,i])
#         
#        perf[method]['avg_survey_mAP'] = avg
#        perf[method]['std_survey_mAP'] = std
# 
#        
#    absc = ['%d-c%d'%(slice_cam/10, slice_cam%10) for slice_cam in slice_cam_id]
#    plt.figure()
#    for i, method in enumerate(method_l):
#        mean_v = perf[method]['avg_survey_mAP'] 
#        std_v = perf[method]['std_survey_mAP']
#        plt.errorbar(absc, mean_v, std_v, linestyle='None', fmt=fmt_l[i],
#                color=color_l[i], alpha=0.25, label=method,
#                capsize=capsize_l[i])
#    fig_fn = 'plots/img/mAP_avg_over_survey_per_slice.png'
#    #plt.title('Slice %d - Camera %d'%(slice_id/10, slice_id%10))
#    plt.legend(loc=0)
#    plt.savefig(fig_fn)
#    plt.close()
#
#    toto = cv2.imread(fig_fn)
#    cv2.imshow('toto', toto)
#    cv2.waitKey(0)



#def perf_mAP_avg_slices():
#    """Average the mAP over slices for each slice. Shows the perf vs the
#    img conditions. """ 
#    # avg over survey for each slice
#    metric = 'mAP'
#    
#    survey_ok = np.ones(8, np.int32)
#    survey_ok[5] = 0
#    survey_ok[7] = 0
#
#    methods_num = len(method_l)
#    for method in method_l:
#        print('method: %s'%method)
#        perf[method]['avg_slice_mAP'] = np.zeros(survey_num)
#        avg = np.zeros(survey_num)
#        std = np.zeros(survey_num)
#
#        for survey_id in range(survey_num):
#            print('survey_id: %d'%survey_id)
#            if survey_id == 8:
#                avg[survey_id] = np.mean(perf[method][metric][survey_id, survey_ok==1])
#                std[survey_id] = np.std(perf[method][metric][ survey_id, survey_ok==1])
#            else:
#                avg[survey_id] = np.mean(perf[method][metric][survey_id,:])
#                std[survey_id] = np.std(perf[method][metric][survey_id,:])
#        perf[method]['avg_slice_mAP'] = avg
#        perf[method]['std_slice_mAP'] = std
#        
#    absc = np.arange(survey_num)
#    plt.figure()
#    for i, method in enumerate(method_l):
#        mean_v = perf[method]['avg_slice_mAP'] 
#        std_v = perf[method]['std_slice_mAP']
#        plt.errorbar(absc, mean_v, std_v, linestyle='None', fmt=fmt_l[i],
#                color=color_l[i], alpha=0.25, label=method,
#                capsize=capsize_l[i])
#    fig_fn = "%s/mAP_avg_over_slice.png"%plot_dir
#    #plt.title('Slice %d - Camera %d'%(slice_id/10, slice_id%10))
#    plt.legend()
#    plt.savefig(fig_fn)
#    plt.close()
#
#    toto = cv2.imread(fig_fn)
#    cv2.imshow('toto', toto)
#    cv2.waitKey(0)
#
