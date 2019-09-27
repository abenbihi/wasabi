"""Evaluates DELF on CMU-Seasons and Symphony image retrieval.

Variation of delf/examples/extract_features.py
"""


def describe_img(args, sess, extractor_fn, centroids, img):
    """Computes DELF local features and VLAD-aggregates them.

    Args:
        args: 
        sess:
        extractor_fn: DELF local features extractor.
        centroids: (NW, DW) np array. NW: Number of visual Words. DW: Dimension
            of visual Words.
        img: (h,w,3) image to describe.
    
    Returns:
        des: global img descriptor of dimension NW*DW.
    """
    (locations_out, descriptors_out, feature_scales_out,
            attention_out) = extractor_fn(im)
    des = lf2vlad(descriptors_out, centroids, args.vlad_norm)
    return des


def describe_survey(args, sess, extractor_fn, centroids, survey):
    """Computes netvlad descriptor for all image of provided survey.

    Args:
        args: various user parameters.
        sess: tf session.
        extractor_fn: DELF local features extractor.
        centroids: (NW, DW) np array. NW: Number of visual Words. DW: Dimension
            of visual Words.
        survey: Survey object.
    
    Returns:
        des_v: np array (survey_size,NW*DW) with one image descriptor per line.
    """
    NW, DW = centroids.shape[:2] # Number of Words, Dim of Words
    des_dim = NW * DW
    survey_size = survey.get_size()
    des_v = np.empty((survey_size, des_dim))
    for idx in range(survey.get_size()):
        if idx % 50 == 0:
            print("%d/%d"%(idx, survey.get_size()))
        img = survey.get_img(idx, proc=False)
        des_v[idx,:] = describe_img(args, sess, extractor_fn, centroids, img)
    return des_v


def bench(args, kwargs, sess, extractor_fn, n_values):
    """Runs and evaluate retrieval on all specified slices and surveys.
        Load surveys, computes descriptor for all images, runs the retrieval and
        compute the metrics.

    Args:
        args: various user parameters.
        kwargs: params to init survey.
        sess: tf session.
        img_op: image placeholder.
        des_op: descriptor operation.
        n_values: list of values to compute the recall at.
    """
    res_dir = "res/netvlad/%d/retrieval/"%args.trial
    perf_dir = "res/netvlad/%d/perf/"%args.trial
    
    retrieval_instance_l = [l.split("\n")[0].split() for l in
            open(args.instance).readlines()]

    for l in retrieval_instance_l:
        # load db traversal
        slice_id = int(l[0])
        cam_id = int(l[1])
        surveyFactory = datasets.survey.SurveyFactory()
        meta_fn = "%s/%d/c%d_db.txt"%(args.meta_dir, slice_id, cam_id)
        #kwargs = {"meta_fn": meta_fn, "img_dir": args.img_dir, "seg_dir": args.seg_dir}
        kwargs["meta_fn"] = meta_fn
        db_survey = surveyFactory.create(args.data, **kwargs)

        for survey_id in l[2:]:
            global_start_time = time.time()
            survey_id = int(survey_id)
            print("\nSlice %d\tCam %d\tSurvey %d"%(slice_id, cam_id, survey_id))
            # check if this bench already exists
            perf_dir = 'res/netvlad/%d/perf'%args.trial
            mAP_fn = "%s/%d_c%d_%d_mAP.txt"%(perf_dir, slice_id, cam_id,
                survey_id)
            recalls_fn = "%s/%d_c%d_%d_rec.txt"%(perf_dir, slice_id, cam_id,
                survey_id)
            if os.path.exists(mAP_fn):
                continue

            # load query traversal
            meta_fn = "%s/%d/c%d_%d.txt"%(args.meta_dir, slice_id, cam_id, 
                    survey_id)
            kwargs["meta_fn"] = meta_fn
            q_survey = surveyFactory.create(args.data, **kwargs)

            # retrieval instance
            retrieval = datasets.retrieval.Retrieval(db_survey, q_survey, args.dist_pos)
            q_survey = retrieval.get_q_survey()


            # describe db img
            local_start_time = time.time()
            db_des_fn = '%s/%d_c%d_db.txt'%(res_dir, slice_id, cam_id)
            if not os.path.exists(db_des_fn): # if you did not compute the db already
                print('** Compute des for database img **')
                db_des_v = describe_survey(args, sess, extractor_fn, centroids,
                        db_survey)
                np.savetxt(db_des_fn, db_des_v)
            else: # if you already computed it, load it from disk
                print('** Load des for database img **')
                db_des_v = np.loadtxt(db_des_fn)
            duration = (time.time() - local_start_time)
            print('(END) run time: %d:%02d'%(duration/60, duration%60))

            # describe q img
            local_start_time = time.time()
            q_des_fn = '%s/%d_c%d_%d.txt'%(res_dir, slice_id, cam_id,
                    survey_id)
            if not os.path.exists(q_des_fn): # if you did not compute the db already
                print('\n** Compute des for query img **')
                q_des_v = describe_survey(args, sess, extractor_fn, centroids,
                        q_survey)
                np.savetxt(q_des_fn, q_des_v)
            else: # if you already computed it, load it from disk
                print('\n** Load des for query img **')
                q_des_v = np.loadtxt(q_des_fn)
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
            recalls = metrics.recallN(order, gt_idx_l, n_values)
            
            duration = (time.time() - local_start_time)
            print('(END) run time: %d:%02d'%(duration/60, duration%60))

  
            # log
            print("\nmAP: %.3f"%mAP)
            for i, n in enumerate(n_values):
                print("Recall@%d: %.3f"%(n, recalls[i]))
            duration = (time.time() - global_start_time)
            print('Global run time retrieval: %d:%02d'%(duration/60, duration%60))
            
            # write retrieval
            order_fn = "%s/%d_c%d_%d_order.txt"%(res_dir, slice_id, cam_id,
                    survey_id)
            rank_fn = "%s/%d_c%d_%d_rank.txt"%(res_dir, slice_id, cam_id,
                    survey_id)
            retrieval.write_retrieval(order, args.top_k, order_fn, rank_fn)

            # write perf
            perf_dir = 'res/netvlad/%d/perf'%args.trial
            np.savetxt(recalls_fn, np.array(recalls))
            np.savetxt(mAP_fn, np.array([mAP]))

def main():
    """Generates graph."""
    # Parse DelfConfig proto.
    config = delf_config_pb2.DelfConfig()
    with tf.gfile.FastGFile(cmd_args.config_path, 'r') as f:
        text_format.Merge(f.read(), config)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        ## Reading list of images.
        #filename_queue = tf.train.string_input_producer(fn_v, shuffle=False)
        #reader = tf.WholeFileReader()
        #_, value = reader.read(filename_queue)
        #image_tf = tf.image.decode_jpeg(value, channels=3)

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            extractor_fn = extractor.MakeExtractor(sess, config)

            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            bench(args, kwargs, sess, extractor_fn, n_values)
