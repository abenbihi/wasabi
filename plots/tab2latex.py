"""Useful primitives to write tabular results to latex."""
import numpy as np

def cmu2latex(latex_fn, all_perf, metrics_name, slice_v, cam_v):
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
