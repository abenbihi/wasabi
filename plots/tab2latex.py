"""Useful primitives to write tabular results to latex."""
import argparse
import os

import numpy as np

def perf2latex(latex_fn, all_perf, metrics_name, slice_v, cam_v):
    """Writes tabular metrics in latex format."""
    slice_num = slice_v.shape[0]

    f = open('%s'%latex_fn, 'w')
    f.write("\\documentclass{article}\n")
    f.write("\\usepackage[utf8]{inputenc}\n")
    f.write("\\usepackage{booktabs} \n")
    f.write("\\usepackage[]{float}\n")
    f.write("\\usepackage[margin=1.2in]{geometry}\n")
    f.write("\\begin{document}\n\n")

    for m_name in metrics_name:
        print(m_name)
        f.write('\\begin{table}[tbh]\n')
        f.write('\\begin{center}\n')
        f.write('\\begin{tabular}{|*{%d}{c|}}\n'%(slice_num + 1))
        f.write('\\hline\n')
        f.write(' Survey')
        #for slice_idx, slice_id in enumerate(slice_cam_id[:-1]):
        for j, (slice_id, cam_id) in enumerate(zip(slice_v, cam_v)):
            f.write(' & %d\_c%d'%(slice_id, cam_id))
        f.write(' \\\\ \n')
        f.write('\\hline\n')

        m = all_perf[m_name]
        print(m.shape)
        survey_num = m.shape[0]
        for i in range(survey_num):
            f.write('%d'%(i))
            for j in range(slice_num):
                f.write(' & %.3f'%m[i,j])
            f.write(' \\\\ \n')

        f.write('\\hline\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{center}\n')
        f.write('\\caption{Metric: %s}\n'%(m_name))
        f.write('\\end{table}\n\n\n')

    f.write('\\end{document}\n')
    print('\\end{document}\n')
    
    f.close()


def log_cmu(perf_dir, slice_v, cam_v, survey_num, metrics_name, n_values,
        latex_fn):
    """ """
    all_perf = {}
    for m_name in metrics_name:
        all_perf[m_name] = -1 * np.ones((survey_num, 2*slice_num))

    for j, (slice_id, cam_id) in enumerate(zip(slice_v, cam_v)):
        for survey_id in range(survey_num):
            if slice_id in [24, 25]:
                if (cam_id == 1) and (survey_id==8): # no ground-truth camera
                    continue
            
            mAP_fn = '%s/perf/%d_c%d_%d_mAP.txt'%(res_dir, slice_id,
                    cam_id, survey_id)
            if os.path.exists(mAP_fn):
                mAP = np.loadtxt(mAP_fn)
                all_perf['mAP'][survey_id, j] = mAP

            recalls_fn = '%s/perf/%d_c%d_%d_rec.txt'%(res_dir,
                    slice_id, cam_id, survey_id)    
            if os.path.exists(recalls_fn):
                recalls = np.loadtxt(recalls_fn)
                for i, m_name in enumerate(metrics_name[1:]):
                    all_perf[m_name][survey_id, j] = recalls[i]       

    for m_name in metrics_name:
        np.savetxt("%s/%s.txt"%(res_dir, m_name), all_perf[m_name], fmt='%.4f')

    # make latex tab
    perf2latex(latex_fn, all_perf, metrics_name, slice_v, cam_v)


if __name__=='__main__':
    n_values = [1, 5, 10, 20]
    metrics_name = ["mAP"] + ["rec%d"%n for n in n_values]

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--data", type=str, required=True, 
            help="{cmu_park, cmu_urban, symphony}")
    args = parser.parse_args()

    if args.data == "cmu_park":
        slice_v = np.array([22, 23, 24, 25])
        cam_v = np.array([0, 1])
        survey_num = 10
    elif args.data == "cmu_urban":
        slice_v = np.array([6, 7, 8])
        cam_v = np.array([0, 1])
        survey_num = 10
    elif args.data == "symphony":
        slice_v = np.array([0])
        cam_v = np.array([0])
        survey_num = 10
    else:
        raise ValueError("I don't know this data: %s"%args.data)

    slice_num = slice_v.shape[0]
    slice_v = np.tile(slice_v.reshape(slice_num,1), (1,2)).flatten()
    cam_v = np.tile(cam_v, (slice_num))
    res_dir = "res/%s/%d/"%(args.method, args.trial)
    latex_fn = "%s/%d.tex"%(res_dir, args.trial)

    log_cmu(res_dir, slice_v, cam_v, survey_num, metrics_name, n_values,
        latex_fn)



