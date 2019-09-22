"""Defines a retrieval instance and metrics."""
import argparse
import os

import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

import datasets.survey as survey
import tools.cst as cst
import tools.metrics as metrics


class Retrieval(object):
    """Creates a retrieval instance.

        A retrieval instance is a pair of image sequence together with their
        camera pose. The first sequence is the database/reference traversal. 
        The second one is the query traversal. The retrieval consists in
        finding the database image nearest to the query one.
    """

    def __init__(self, db_survey, q_survey, dist_pos):
        """Creates a retrieval instance from two surveys.

        Computes the matching image pairs between the database and the query
        survey. A pair of images match if their camera euclidean distance is
        below dist_pos.
        
        Args:
            db_survey: database survey
            q_survey: query survey
            dist_pos: distance between positives
        """
        self.dist_pos = dist_pos
        self.db_survey = db_survey
        self.q_survey = q_survey
        
        self.db_size = self.db_survey.get_size()
        self.db_fn_v = self.db_survey.get_fn_v()
        self.db_pose_v = self.db_survey.get_pose_v()
        self.q_size = self.q_survey.get_size()
        self.q_fn_v = self.q_survey.get_fn_v()
        self.q_pose_v = self.q_survey.get_pose_v()

        self.gt_idx_l = None
        self.gt_name_d = None
        self.get_gt_idx()
        self.get_gt_mAP()
        
        # test to see if I correctly filter out the query without matching db
        aa = len(self.gt_idx_l)
        ab = self.q_fn_v.shape[0]
        ac = len(list(self.gt_name_d.keys()))
        if (aa!=ab) or (aa!=ac) or (ab!=ac):
            raise ValueError("You did not filter out the useless queries correctly.")

    def get_img_q(self, idx):
        """Returns idx-th image of query survey."""
        return self.q_survey.get_img(idx)

    def get_img_db(self, idx):
        """Returns idx-th image of database survey."""
        return self.db_survey.get_img(idx)

    def get_pose_q(self, idx):
        """Returns idx-th pose of query survey."""
        return self.q_survey.get_pose(idx)

    def get_pose_db(self, idx):
        """Returns idx-th of database survey."""
        return self.db_survey.get_pose(idx)

    def get_gt_idx(self, dist_pos=None):
        """Computes pairs of matching database-query image indices.

        self.gt_idx_l[idx] is the list of database image indices matching the
        idx-th query image.

        Args:
            dist_pos: distance between positives i.e. matching img.
        """
        if self.gt_idx_l is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.db_pose_v)
            positives = knn.radius_neighbors(self.q_pose_v,
                    radius=self.dist_pos, return_distance=False)

            # filter out queries that don't have matching db images
            self.gt_idx_l = []
            q_ok = np.ones(self.q_size, np.uint8)
            for i, gt_v in enumerate(positives):
                if gt_v.size==0:
                    #print('Query %d has not matching db img, remove it'%i)
                    q_ok[i] = 0
                else:
                    self.gt_idx_l.append(gt_v)

            # update query survey
            self.q_survey.update(q_ok)
            self.q_fn_v = self.q_survey.get_fn_v()
            self.q_size  = self.q_survey.get_size()
        return self.gt_idx_l

    def get_gt_mAP(self, dist_pos=None):
        """Computes pairs of matching database-query image filenames.
        
        self.gt_name_d[idx] is the list of database image names matching the
        idx-th query image.
        """
        if self.gt_name_d is None:
            if self.gt_idx_l is None: # compute matching pair indices
                self.get_gt_idx(self.dist_pos)
            self.gt_name_d = {}
            for i, q_fn in enumerate(self.q_fn_v):
                self.gt_name_d[q_fn] = self.db_fn_v[self.gt_idx_l[i]]
        return self.gt_name_d

    def get_gt_rank(self, mode):
        """Returns ground-truth retrieval rank."""
        if mode == "name": # returns names of ground-truth retrieval
            return self.gt_name_d
        elif mode == "idx": # returns idx of ground-truth retrieval
            return self.gt_idx_l
        else:
            raise ValueError("Uknown mode: %s"%mode)

    def show_retrieval(self):
        """Shows matching img pairs."""
        for q_idx in range(self.q_size):
            q_img = self.q_survey.get_img(q_idx)
            for db_idx in self.gt_idx_l[q_idx]:
                db_img = self.db_survey.get_img(db_idx)
                cv2.imshow('q | db', np.hstack((q_img, db_img)))
                if (cv2.waitKey(0) & 0xFF) == ord("q"):
                    exit(0)

    def random_retrieval(self):
        """Randomly match q and db for random retrieval baseline."""
        if self.gt_idx_l is None:
            self.get_gt_idx()

        self.order_l = []
        for _ in range(self.q_size):
            order = np.arange(self.db_size)
            np.random.shuffle(order)
            self.order_l.append(order)
        return self.order_l

    def get_retrieval_rank(self, order_l, top_k):
        """Format the retrieval to feed it to the mAP function.

        The mAP metric code expects for each query, a string line in the format 
        "%s [%d %s]*"%(q_fn, [rank, db_fn]*). Instead I provide the split of
        this line. The mAP code is adapted accordingly.
        """
        input_mAP_l = [] 
        for i, gt_idx_v in enumerate(self.gt_idx_l):
            tmp_l = []
            tmp_l.append(self.q_fn_v[i])

            order = order_l[i]
            gt_rank = metrics.get_order(order, gt_idx_v)
            # write top_k retrieved images
            for ii,jj in enumerate(order[:top_k]):
                tmp_l.append("%d"%ii)
                tmp_l.append("%s"%self.db_survey.fn_v[jj])

            # write rank and filenames of matching images outside of the top_k
            missing = np.where(gt_rank>top_k)[0]
            for jj in missing:
                tmp_l.append("%d"%jj)
                tmp_l.append("%s"%self.db_survey.fn_v[jj])
            input_mAP_l.append(tmp_l)
        return input_mAP_l

    def write_retrieval(self, order_l, top_k, order_fn, rank_fn):
        """ Write retrieval to file.
        
        Args:
            order_l: list of arrays. order_l[i] is the list of database image
                indices for nearest to furthest in descriptor space.
            top_k: number of images to retrieve.
            order_fn: file to write order_l to.
            rank_fn: file to write top_k image filenames for each query.
        """
        np.savetxt(order_fn, np.vstack(order_l), fmt='%d')
        
        f_out = open(rank_fn, 'w')
        for i, gt_idx_v in enumerate(self.gt_idx_l):
            order = order_l[i]
            gt_rank = self.get_order(order, gt_idx_v)
            f_out.write(self.q_survey.fn_v[i])
            # write top_k retrieved images
            for ii,jj in enumerate(order[:top_k]):
                f_out.write(' %d %s'%(ii, self.db_survey.fn_v[jj]))
            # write rank and filenames of matching images outside of the top_k
            missing = np.where(gt_rank>top_k)[0]
            for jj in missing:
                f_out.write(' %d %s'%(jj, self.db_survey.fn_v[jj]))
            f_out.write('\n')
        f_out.close()

def test_show_retrieval():
    """Test retrieval code. Shows matching images."""
    data = "cmu"
    meta_dir = "meta/%s/surveys"%data
    if data == "cmu":
        dist_pos = 5.
        img_dir = "%s/datasets/Extended-CMU-Seasons/"%cst.WS_DIR
        seg_dir = "%s/tf/cross-season-segmentation/res/ext_cmu/"%cst.WS_DIR 
        slice_id, cam_id, survey_id = 22, 0, 0

        surveyFactory = survey.SurveyFactory()
        meta_fn = "%s/%d/c%d_db.txt"%(meta_dir, slice_id, cam_id)
        kwargs = {"meta_fn": meta_fn, "img_dir": img_dir, "seg_dir": seg_dir}
        db_survey = surveyFactory.create("cmu", **kwargs)
        
        meta_fn = "%s/%d/c%d_%d.txt"%(meta_dir, slice_id, cam_id, survey_id)
        kwargs["meta_fn"] = meta_fn
        q_survey = surveyFactory.create("cmu", **kwargs)

        retrieval = Retrieval(db_survey, q_survey, dist_pos)
        retrieval.show_retrieval()


if __name__=='__main__':
    test_show_retrieval()

