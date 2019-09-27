"""Convert delf codebooks from tf format to np format."""
import argparse
import os

import numpy as np
import tensorflow as tf

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True, 
            help="checkpoint dir")
    parser.add_argument("--out", type=str, required=True, help="output file")
    args = parser.parse_args()

    ## OK
    #ckpt = 'meta/words/delf/rparis6k_codebook_1024/k1024_codebook_tfckpt/codebook'
    #print(ckpt)
    #new_saver = tf.train.import_meta_graph('meta/words/delf/rparis6k_codebook_1024/k1024_codebook_tfckpt/codebook.meta')
    
    # I do this to avoid '//' in the ckpt_dir. It makes tf crash.
    dirname = os.path.dirname(args.ckpt_dir)
    ckpt = "%s/codebook"%dirname
    new_saver = tf.train.import_meta_graph("%s/codebook.meta"%dirname)

    ## get all feature maps names (debug)
    #a = [n.name for n in tf.get_default_graph().as_graph_def().node]
    #for aa in a:
    #    print(aa)
    feat_name, tensor_name = 'clusters', 'clusters:0'

    with tf.Session() as sess:
        new_saver.restore(sess, ckpt)
        feat_op = sess.graph.get_tensor_by_name(tensor_name)
        #print(feat_op.get_shape())
        clusters = sess.run(feat_op)
        #print(clusters.shape)
        np.savetxt(args.out, clusters)

