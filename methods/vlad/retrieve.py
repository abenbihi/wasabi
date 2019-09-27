"""VLAD implementation.

Variation of https://github.com/jorjasso/VLAD
"""
import argparse
import os
import pickle
import time

import cv2
import numpy as np

import datasets.survey
import datasets.retrieval
from tools import metrics

class OpenCVFeatureExtractor(object):
    """Local feature extractor using OpenCV methods."""

    def __init__(self):
        """Instantiates a local feature extractor."""
    
    def get_local_features(self, img):
        """Detect and describe local keypoints.

        Returns:
            des_v: (N, D) np array of N local descriptors of dimension D.
        """
        _, des =self.fe.detectAndCompute(img, None)
        return des

class SIFTFeatureExtractor(OpenCVFeatureExtractor):
    """Local feature extractor using OpenCV methods."""

    def __init__(self, max_num_feat):
        if args.max_num_feat != -1:
            self.fe = cv2.xfeatures2d.SIFT_create(args.max_num_feat)
        else:
            self.fe = cv2.xfeatures2d.SIFT_create()

class SURFFeatureExtractor(OpenCVFeatureExtractor):
    """Local feature extractor using OpenCV methods."""

    def __init__(self):
        self.fe = cv2.xfeatures2d.SURF_create(400)

class ORBFeatureExtractor(OpenCVFeatureExtractor):
    """Local feature extractor using OpenCV methods."""

    def __init__(self, max_num_feat):
        self.fe = cv2.ORB_create()

class MSERFeatureExtractor(OpenCVFeatureExtractor):
    """Local feature extractor using OpenCV methods."""

    def __init__(self, max_num_feat):
        self.fe = cv2.MSER_create()

class AKAZEFeatureExtractor(OpenCVFeatureExtractor):
    """Local feature extractor using OpenCV methods."""

    def __init__(self, max_num_feat):
        self.fe = cv2.AKAZE_create()
   
class DelfFeatureExtractor(object):
    """Object to read computed delf local features from disk."""

    def __init__(self, args):
        """Instantiates a local feature extractor."""
        self.des_dir = args.delf_dir


    def get_local_features(self, img_fn):
        """ """

class FeatureExtractorFactory(object):
    """Provides one of the registered local feature extractor."""

    def __init__(self):
        self._builders = {}
        self._builders["sift"] = SIFTFeatureExtractor
        self._builders["surf"] = SURFFeatureExtractor
        self._builders["orb"] = ORBFeatureExtractor
        self._builders["mser"] = MSERFeatureExtractor
        self._builders["akaze"] = AKAZEFeatureExtractor
        self._builders["delf"] = DelfFeatureExtractor

    def register_builder(self, key, builder):
        self._builders[key] = builder

    def create(self, key, kwargs):
        builder = self._builders[key]
        if not builder:
            raise ValueError("Unknown feature extractor: %s"%key)
        return builder(**kwargs)


def fvecs_read(filename, c_contiguous=True):
    """Reads the fvecs format. Returns np array.
    
    Copied from: https://gist.github.com/danoneata/49a807f47656fedbb389
    """
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def load_cbk_Flickr100k():
    """Returns vlad codebook trained on Flicker100k."""
    return fvecs_read('meta/words/holidays/clust_k64.fvecs')

def load_cbk_delf_par1024():
    """Returns delf codebook trained on Paris6k with 1024 centroids."""
    return np.loadtxt("meta/k1024_paris.txt")

class Codebook():
    """Loads various codebook.
        
        NW: Number of visual Words.
        DW: Dimension of visual Words.
    """
    def __init__(self):
        self.d = {}
        # vlad # (NW, DW) = (64,128)
        self.d["flickr100k"] = load_cbk_Flickr100k

        # delf
        self.d["delf_k1024_paris"] = load_cbk_delf_par1024


    
    def load(self, key):
        loader = self.d[key]
        if not loader:
            raise ValueError("Unknown codebook: %s"%key)
        return loader()


def gen_codebook():
    """ """


def normalise_vlad(des_vlad, vlad_norm='l2'):
    """
    Flatten and normalise a vlad descriptor.
    Args:
        des_vlad: (vlad_dim,)
        vlad_norm: {l2, ssr}
    """
    if vlad_norm=='l2':
        des_vlad = des_vlad.flatten()
        norm = np.sqrt(np.sum(des_vlad**2))
        if norm > 1e-8:
            des_vlad /= np.sqrt(np.sum(des_vlad**2))
        else:
            des_vlad = np.ones(des_vlad.size)
    elif vlad_norm=='ssr':
        des_vlad = des_vlad.flatten()
        # power normalization, also called square-rooting normalization
        des_vlad = np.sign(des_vlad)*np.sqrt(np.abs(des_vlad))
        # L2 normalization
        des_vlad = des_vlad/np.sqrt(np.dot(des_vlad,des_vlad))
    else:
        raise ValueError("Unknown norm: %s"%vlad_norm)
    return des_vlad


def lf2vlad(lf, centroids, vlad_norm):
    """Computes vlad descriptors from the local features.
    Args:
        lf: (N, D) np array. N Local Features of dimension D.
        centroids: (NW, DW) np array. NW visual words of dimension DW.
    """
    NW, DW = centroids.shape[:2]
    N, D = lf.shape[:2] # dimension of local img feature
    if DW != D:
        raise ValueError("Error: D != DW."
                "Local descriptor dim is different than the visual word's one.")

    # cluster assignment
    c = np.expand_dims(centroids, 1) # row
    x = np.expand_dims(lf, 0) # col
    # euclidean distance between all pairs of local descriptors and visual
    # words/cluster
    d = np.linalg.norm(c - x, ord=None, axis=2) 
    x2c = np.argmin(d, axis=0) # x2c[j] = cluster of j-th local descriptor
    # one hot encoding of the cluster assignment
    x2c_hot = np.eye(NW)[x2c].T # x2c_host[i,j]=1 if x_j belongs to c_i

    # cluster distance
    gap = x - c # gap[i,j] = c_i - x_j

    # sum of eq 1 of vlad paper: 
    # aggregating local descriptors into a compact image representation
    toto = gap * np.expand_dims(x2c_hot, 2)
    v = np.sum(toto, axis=1)
    
    # normalisation
    v = normalise_vlad(v, vlad_norm)
    return v


def describe_img(args, fe, img):
    """Computes a global descriptor for the image.

    Returns:
        des: (1, D) a global image descriptor of dimension D for img.
    """
    lf = fe.get_local_features(img)
    if args.agg_mode == "vlad":
        des = lf2vlad(lf, centroids, args.vlad_norm)
    elif args.agg_mode == "bow":
        des = lf2bow(lf, centroids, args.bow_norm)
    else:
        raise ValueError("Unknown aggregation mode: %s"%args.agg_mode)
    return des


def describe_survey(args, fe, centroids, survey):
    """ """
    NW, DW = centroids.shape[:2] # Number of Words, Dim of Words
    des_dim = NW * DW # global img descriptor dimension
    survey_size = survey.get_size()
    des_v = np.empty((survey_size, des_dim))
    for idx in range(survey.get_size()):
        if idx % 50 == 0:
            print("%d/%d"%(idx, survey.get_size()))
        img = survey.get_img(idx, proc=False)
        des_v[idx,:] = describe_img(args, fe, img)
    return des_v


def bench(args, kwargs, centroids, n_values):
    """ """
    global_start_time = time.time()

    # check if this bench already exists
    perf_dir = "res/%s/%d/perf"%(args.agg_mode, args.trial)
    mAP_fn = "%s/%d_c%d_%d_mAP.txt"%(perf_dir, args.slice_id, args.cam_id,
        args.survey_id)
    recalls_fn = "%s/%d_c%d_%d_rec.txt"%(perf_dir, args.slice_id, args.cam_id,
        args.survey_id)
    if os.path.exists(mAP_fn):
        return -1, -1 
    
    res_dir = "res/%s/%d/retrieval/"%(args.agg_mode, args.trial)

    # load db traversal
    surveyFactory = datasets.survey.SurveyFactory()
    meta_fn = "%s/%d/c%d_db.txt"%(args.meta_dir, args.slice_id, args.cam_id)
    kwargs["meta_fn"] = meta_fn
    db_survey = surveyFactory.create(args.data, **kwargs)
    
    # load query traversal
    meta_fn = "%s/%d/c%d_%d.txt"%(args.meta_dir, args.slice_id, args.cam_id, 
            args.survey_id)
    kwargs["meta_fn"] = meta_fn
    q_survey = surveyFactory.create(args.data, **kwargs)

    # retrieval instance. Filters out queries without matches.
    retrieval = datasets.retrieval.Retrieval(db_survey, q_survey, args.dist_pos)
    q_survey = retrieval.get_q_survey()

    # choose a local feature extractor
    feFactory = FeatureExtractorFactory()
    kwargs = {"max_num_feat": args.max_num_feat}
    fe = feFactory.create(args.lf_mode, kwargs)

    # describe db img
    local_start_time = time.time()
    db_des_fn = '%s/%d_c%d_db.pickle'%(res_dir, args.slice_id, args.cam_id)
    if not os.path.exists(db_des_fn): # if you did not compute the db already
        print('** Compute des for database img **')
        db_img_des_v = describe_survey(args, fe, centroids, db_survey)
        with open(db_des_fn, 'wb') as f:
            pickle.dump(db_img_des_v, f)
    else: # if you already computed it, load it from disk
        print('** Load des for database img **')
        with open(db_des_fn, 'rb') as f:
            db_img_des_v = pickle.load(f)
    duration = (time.time() - local_start_time)
    print('(END) run time: %d:%02d'%(duration/60, duration%60))


    # describe q img
    local_start_time = time.time()
    q_des_fn = '%s/%d_c%d_%d.pickle'%(res_dir, args.slice_id, args.cam_id,
            args.survey_id)
    if not os.path.exists(q_des_fn): # if you did not compute the db already
        print('\n** Compute des for query img **')
        q_img_des_v = describe_survey(args, fe, centroids, q_survey)
        with open(q_des_fn, 'wb') as f:
            pickle.dump(q_img_des_v, f)
    else: # if you already computed it, load it from disk
        print('\n** Load des for database img **')
        with open(q_des_fn, 'rb') as f:
            q_img_des_v = pickle.load(f)
    duration = (time.time() - local_start_time)
    print('(END) run time: %d:%02d'%(duration/60, duration%60))
    
    
    # retrieve each query
    print('\n** Retrieve query image **')
    local_start_time = time.time()
    d = np.linalg.norm(np.expand_dims(q_img_des_v, 1) -
            np.expand_dims(db_img_des_v, 0), ord=None, axis=2)
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
    order_fn = "%s/%d_c%d_%d_order.txt"%(res_dir, args.slice_id, args.cam_id,
            args.survey_id)
    rank_fn = "%s/%d_c%d_%d_rank.txt"%(res_dir, args.slice_id, args.cam_id,
            args.survey_id)
    retrieval.write_retrieval(order, args.top_k, order_fn, rank_fn)

    # write perf
    perf_dir = 'res/vlad/%d/perf'%args.trial
    np.savetxt(recalls_fn, np.array(recalls))
    np.savetxt(mAP_fn, np.array([mAP]))
    return mAP, recalls


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=int, required=True, help='Trial.')
    parser.add_argument('--dist_pos', type=float, required=True)
    parser.add_argument('--top_k', type=int, default=20)

    parser.add_argument("--lf_mode", type=str, 
            help="Local feature extraction method {sift, surf, orb, akaze, delf}")
    parser.add_argument('--max_num_feat', type=int)
    parser.add_argument("--agg_mode", type=str, 
            help="Aggregation mode {bow, vlad}")
    parser.add_argument('--vlad_norm', type=str)
    parser.add_argument('--n_words', type=int)
    parser.add_argument('--centroids', type=str)
    
    parser.add_argument('--data', type=str, required=True, help='{cmu, lake}')
    parser.add_argument('--slice_id', type=int, default=22)
    parser.add_argument('--cam_id', type=int, default=0)
    parser.add_argument('--survey_id', type=int, default=0)

    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--meta_dir', type=str, required=True)
    parser.add_argument('--mask_dir', type=str, default="")
    
    parser.add_argument('--resize', type=int, default=0, help='set to 1 to resize img')
    parser.add_argument('--h', type=int, default=480, help='new height')
    parser.add_argument('--w', type=int, default=704, help='new width')
    args = parser.parse_args()

    n_values = [1, 5, 10, 20]
 
    res_dir = "res/%s/%d/retrieval/"%(args.agg_mode, args.trial)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    perf_dir = "res/%s/%d/perf/"%(args.agg_mode, args.trial)
    if not os.path.exists(perf_dir):
        os.makedirs(perf_dir)

    if args.data == "cmu":
        kwargs = {"img_dir": args.img_dir, "seg_dir": ""}
    elif args.data == "lake":
        kwargs = {"img_dir": args.img_dir, "seg_dir": "", 
                "mask_dir": args.mask_dir}
    else:
        raise ValueError("I don't know this dataset: %s"%args.data)


    codebook = Codebook()
    centroids = codebook.load(args.centroids)

    bench(args, kwargs, centroids, n_values)
