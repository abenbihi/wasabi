"""Evaluate NetVLAD on CMU-Seasons and Symphony image retrieval."""
import os, argparse, time

import cv2
import numpy as np

import tensorflow as tf

import datasets.survey
import datasets.retrieval
from tools import metrics

import third_party.tf_land.netvlad.tools.model_netvlad as netvlad


# weights of model pre-trained on Pittsburg dataset


def describe_survey(args, sess, des_op, survey):
    """Computes netvlad descriptor for all image of provided survey.
    
    Returns:
        des_v: np array (survey_size, descriptor dimension) with one image
            descriptor per line.
    """
    mean_std = np.loadtxt(args.mean_fn)
    mean = mean_std[0,:]*255.
    std = mean_std[1,:]*255.

    des_dim = des_op.get_shape().as_list()[-1]
    survey_size = survey.get_size()
    des_v = np.empty((survey_size, des_dim))
    for idx in range(survey.get_size()):
        img = survey.get_img(idx)
        img = cv2.resize(img, (args.w, args.h), interpolation=cv2.INTER_AREA)
        img = (img.astype(np.float32) - mean)/std
        img = np.expand_dims(img, 0)
        db_des_v[db_idx,:] = sess.run(des_op, feed_dict={img_op: img})
    return des_v


def bench(args, sess, des_op, n_values):
    """Runs and evaluate retrieval on all specified slices and surveys."""
    retrieval_instance_l = [int(l.split("\n")[0]) for l in
            open(args.instance).readlines()]

    for l in retrieval_instance_l:
        # load db traversal
        slice_id = l[0]
        cam_id = l[1]
        surveyFactory = datasets.survey.SurveyFactory()
        meta_fn = "%s/%d/c%d_db.txt"%(args.meta_dir, args.slice_id, args.cam_id)
        kwargs = {"meta_fn": meta_fn, "img_dir": args.img_dir, "seg_dir": args.seg_dir}
        db_survey = surveyFactory.create(args.data, **kwargs)

        for survey_id in l[2:]:
            # check if this bench already exists
            perf_dir = 'res/netvlad/%d/perf'%args.trial
            mAP_fn = "%s/%d_c%d_%d_mAP.txt"%(perf_dir, slice_id, cam_id,
                survey_id)
            recalls_fn = "%s/%d_c%d_%d_rec.txt"%(perf_dir, slice_id, cam_id,
                survey_id)
            if os.path.exists(mAP_fn):
                return -1, -1 

            # load query traversal
            meta_fn = "%s/%d/c%d_%d.txt"%(args.meta_dir, args.slice_id, args.cam_id, 
                    args.survey_id)
            kwargs["meta_fn"] = meta_fn
            q_survey = surveyFactory.create(args.data, **kwargs)

            # retrieval instance
            retrieval = datasets.retrieval.Retrieval(db_survey, q_survey, args.dist_pos)
            q_survey = retrieval.get_q_survey()

            # describe db img
            local_start_time = time.time()
            db_des_fn = '%s/%d_c%d_db.pickle'%(res_dir, args.slice_id, args.cam_id)
            if not os.path.exists(db_des_fn): # if you did not compute the db already
                print('** Compute des for database img **')
                db_des_v = describe_survey(args, sess, des_op, db_survey)
                np.savetxt(db_des_fn, db_des_v)
            else: # if you already computed it, load it from disk
                print('** Load des for database img **')
                db_des_v = np.loadtxt(db_des_fn)
            duration = (time.time() - local_start_time)
            print('(END) run time: %d:%02d'%(duration/60, duration%60))

            # describe q img
            local_start_time = time.time()
            q_des_fn = '%s/%d_c%d_%d.pickle'%(res_dir, args.slice_id, args.cam_id,
                    args.survey_id)
            if not os.path.exists(q_des_fn): # if you did not compute the db already
                print('\n** Compute des for query img **')
                q_des_v = describe_survey(args, sess, des_op, q_survey)
                np.savetxt(q_des_fn, q_des_v)
            else: # if you already computed it, load it from disk
                print('\n** Load des for query img **')
                q_des_v = np.loadtxt(db_des_fn)
            duration = (time.time() - local_start_time)
            print('(END) run time: %d:%02d'%(duration/60, duration%60))

            # retrieve each query
            print('\n** Retrieve query image **')
            local_start_time = time.time()
            d = np.linalg.norm(np.expand_dims(q_des_v, 1) -
                    np.expand_dims(db_des_v, 0), ord=None, axis=2)
            order = np.argsort(d, axis=1)
            #np.savetxt(order_fn, order, fmt='%d')
            duration = (time.time() - local_start_time)
            print('(END) run time %d:%02d'%(duration/60, duration%60))
   
            # compute perf
            print('\n** Compute performance **')
            local_start_time = time.time()
            rank_l = retrieval.get_retrieval_rank(order, args.top_k)
            
            gt_name_d = retrieval.get_gt_rank("name")
            mAP = metrics.mAP(rank_l, gt_name_d)

            gt_idx_l = retrieval.get_gt_rank("idx")
            recalls = metrics.recallN(order_l, gt_idx_l, n_values)
            
            duration = (time.time() - local_start_time)
            print('(END) run time: %d:%02d'%(duration/60, duration%60))

  
            # log
            print("\nmAP: %.3f"%mAP)
            for i, n in enumerate(n_values):
                print("Recall@%d: %.3f"%(n, recalls[i]))
            duration = (time.time() - global_start_time)
            print('Global run time retrieval: %d:%02d'%(duration/60, duration%60))
            
            # write retrieval
            order_fn = "%s/order_%d_c%d_%d.txt"%(res_dir, args.slice_id, args.cam_id,
                    args.survey_id)
            rank_fn = "%s/rank_%d_c%d_%d.txt"%(res_dir, args.slice_id, args.cam_id,
                    args.survey_id)
            retrieval.write_retrieval(order_l, args.top_k, order_fn, rank_fn)

            # write perf
            perf_dir = 'res/netvlad/%d/perf'%args.trial
            np.savetxt(recalls_fn, np.array(recalls))
            np.savetxt(mAP_fn, np.array([mAP]))
            return mAP, recalls


def main(args, n_values):
    """Builds the graph and bench the retrieval."""
    with tf.Graph().as_default():
        # descriptor op
        img_op = tf.placeholder(dtype=tf.float32, shape=[1, args.h, args.w, 3])
        des_op = netvlad.vgg16NetvladPca(img_op)

        # set saver to init netvlad with paper weights
        var_to_init = []
        var_to_init_name = list(np.loadtxt("%s/meta/var_to_init_netvlad.txt"
            %args.netvlad_dir, dtype=str))

        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for var in all_vars:
            if var.op.name in var_to_init_name:
                var_to_init.append(var)
        saver_init = tf.train.Saver(var_to_init)
        
        # Set saver to restore finetuned network 
        variable_averages = tf.train.ExponentialMovingAverage(args.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Restore vars
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if args.no_finetuning == 1: # load model from pittsburg trainng
                netvlad_ckpt_dir = "%s/meta/weights/netvlad_tf_open/vd16_pitts30k_conv5_3_vlad_preL2_intra_white"%args.netvlad_dir

                global_step = 0
                print("Evaluate netvlad from the paper")
                ckpt = tf.train.get_checkpoint_state(netvlad_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print("checkpoint path: ", ckpt.model_checkpoint_path)
                    saver_init.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print('No checkpoint file found at: %s'%netvlad_dir)
                    return
                print('Load model Done')
            else:
                print("Evaluate my super finetuned version")
                train_log_dir = 'res/%d/log/train'%args.trial
                ckpt = tf.train.get_checkpoint_state(train_log_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print("checkpoint path: ", ckpt.model_checkpoint_path)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                else:
                    print('No checkpoint file found')
                    return
   
            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                # TODO: check if my code is using this
                threads = [] 
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord,
                        daemon=True, start=True))

                bench(args, sess, des_op, n_values)
            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=int, required=True, help='Trial.')
    parser.add_argument('--dist_pos', type=float, required=True)
    parser.add_argument('--top_k', type=int, default=20)

    parser.add_argument('--data', type=str, required=True, help='{cmu, lake}')
    parser.add_argument('--instance', type=str, required=True)
    
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--seg_dir', type=str, required=True)
    parser.add_argument('--meta_dir', type=str, required=True)

    parser.add_argument('--mean_fn', type=str, default='', help='Path to mean/std.')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--resize', type=int, default=0, help='set to 1 to resize img')
    parser.add_argument('--h', type=int, default=480, help='new height')
    parser.add_argument('--w', type=int, default=704, help='new width')
    parser.add_argument('--moving_average_decay', type=float, default=0.9999, help='')
    parser.add_argument('--no_finetuning', type=int, help='Set to 1 if you start train.')

    parser.add_argument('--netvlad_dir', type=str)
    args = parser.parse_args()
 
 
    res_dir = "res/netvlad/%d/retrieval/"%args.trial
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    perf_dir = "res/netvlad/%d/perf/"%args.trial
    if not os.path.exists(perf_dir):
        os.makedirs(perf_dir)

    n_values = [1, 5, 10, 20]
    main(args, n_values)
