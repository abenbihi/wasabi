"""Bench the random retrieval on CMU Seasons."""
import argparse
import os

import numpy as np

import datasets.survey
import datasets.retrieval
import tools.metrics as metrics
import plots.tab2latex as tab2latex

def random_retrieval(args, retrieval, n_values, write=False):
    """Random retrieval baseline for one survey."""
    mAP_l = []
    recall_l = []
    for _ in range(args.repeat):
        order_l = retrieval.random_retrieval()
        if write:
            res_dir = 'res/random/%d/'%args.trial
            retrieval.write_retrieval(order_l, args.top_k,
                    '%s/order.txt'%res_dir, '%s/rank.txt'%res_dir)

        rank_l = retrieval.get_retrieval_rank(order_l, args.top_k)
        gt_name_d = retrieval.get_gt_rank("name")
        mAP = metrics.mAP(rank_l, gt_name_d)
        mAP_l.append(mAP)

        gt_idx_l = retrieval.get_gt_rank("idx")
        recalls = metrics.recallN(order_l, gt_idx_l, n_values)
        recall_l.append(recalls)

    mAP_avg = np.sum(mAP_l)/args.repeat
    print('mAP: %.3f'%mAP_avg)
    
    recall_avg = np.sum(np.vstack(recall_l), axis=0) / args.repeat
    for i,n in enumerate(n_values):
        print("- Recall@%d: %.4f"%(n, recall_avg[i]))
    return mAP_avg, recall_avg


def bench_random_retrieval_cmu(args, slice_v, cam_v, n_values):
    """Bench radom retrieval on all surveys."""
    metrics_name = ["mAP"] + ["rec%d"%n for n in n_values]
    survey_num = 10
    
    all_perf = {}
    for m_name in metrics_name:
        all_perf[m_name] = -1 * np.ones((survey_num, 2*slice_num))
    
    for j, (slice_id, cam_id) in enumerate(zip(slice_v, cam_v)):
        # load db survey
        surveyFactory = datasets.survey.SurveyFactory()
        meta_fn = "%s/%d/c%d_db.txt"%(args.meta_dir, slice_id, cam_id)
        kwargs = {"meta_fn": meta_fn, "img_dir": args.img_dir, "seg_dir": args.seg_dir}
        db_survey = surveyFactory.create("cmu", **kwargs)

        print('\n** Slice %d\tCam %d **'%(slice_id, cam_id))
        for survey_id in range(survey_num):
            if slice_id in [24, 25]:
                if (cam_id == 1) and (survey_id==8): # no ground-truth camera
                    continue
            # load q survey
            print("Survey: %d"%survey_id)
            meta_fn = "%s/%d/c%d_%d.txt"%(args.meta_dir, slice_id, cam_id, survey_id)
            kwargs["meta_fn"] = meta_fn
            q_survey = surveyFactory.create("cmu", **kwargs)
            retrieval = datasets.retrieval.Retrieval(db_survey, q_survey, args.dist_pos)

            mAP_avg, recall_avg = random_retrieval(args, retrieval, n_values)
            all_perf['mAP'][survey_id, j] = mAP_avg
            for k, m_name in enumerate(metrics_name[1:]):
                all_perf[m_name][survey_id, j] = recall_avg[k]
    
    res_dir = 'res/random/%d'%args.trial
    for m_name in metrics_name:
        np.savetxt('%s/%s.txt'%(res_dir, m_name), all_perf[m_name])
        #all_perf[m_name] = np.loadtxt('%s/%s.txt'%(res_dir, m_name))
    latex_fn = "res/random/%d/%d.tex"%(args.trial, args.trial)
    tab2latex.perf2latex(latex_fn, all_perf, metrics_name, slice_v, cam_v)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=int, required=True, help="Id of this xp")
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--dist_pos', type=float, required=True)
    
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--seg_dir', type=str, required=True)
    parser.add_argument('--meta_dir', type=str, required=True)
    
    parser.add_argument('--repeat', type=int, required=True, 
            help="Number of time you repeat the random retrieval.")
    args = parser.parse_args()

    res_dir = 'res/random/%d'%args.trial
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    n_values = [1, 5, 10, 20]

    cam_v = np.array([0, 1])
    # park
    slice_v = np.array([22, 23, 24, 25])
    slice_num = slice_v.shape[0]
    slice_v = np.tile(slice_v.reshape(slice_num,1), (1,2)).flatten()
    cam_v = np.tile(cam_v, (slice_num))
    bench_random_retrieval_cmu(args, slice_v, cam_v, n_values)

    ## urban
    #trial = args.trial + 1
    #slice_v = np.array([6,7,8])
    #slice_num = slice_v.shape[0]
    #slice_v = np.tile(slice_v.reshape(slice_num,1), (1,2)).flatten()
    #cam_v = np.tile(cam_v, (slice_num))
    #bench_random_retrieval_cmu(args, slice_v, cam_v, n_values)

