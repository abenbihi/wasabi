"""Defines Survey class."""
import os, argparse, time
import numpy as np
import cv2

import tools.cst as cst
import tools.semantic_proc as semantic_proc

class Survey(object):
    """Base class to manipulate a survey/traversal.
        
       A survey (sometimes called traversal) is a sequence of sucessive images
       together with their pose of a location.

        Attributes:
            img_dir: bgr image directory.
            fn_v: list of image filenames with respect to img_dir.
            pose_v: list of camera pose associated to each image.
    """
    
    def __init__(self, meta_fn, img_dir, seg_dir):
        """Creates a new Survey.

        This must be called by the constructors of subclasses.

        Args:
            img_dir: bgr image directory.
            seg_dir: semantic segmentation image directory.
            meta_fn: file with list of image names and poses, named:
                "%d/c%d_%d.txt"%(slice_id, cam_id, survey_id).
                Each line is formatted as follow: img name relative to img_dir,
                camera rotation in quaternion (qw, qx, qy, qz), camera 
                translation (x,y,z) and not camera center.

        """
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        
        if os.stat(meta_fn).st_size == 0:
            raise ValueError("This file is empty: %s"%meta_fn)

        meta = np.loadtxt(meta_fn, dtype=str)
        self.fn_v = meta[:,0]

        self.pose_v = meta[:,5:].astype(np.float32)
        self.size = self.fn_v.shape[0]
        self.survey_ok = 1

    def get_size(self):
        """Returns the size of the survey i.e. the number of images."""
        return self.size

    def get_fn_v(self):
        """Returns the array of filenames."""
        return self.fn_v

    def get_pose_v(self):
        """Returns the array of camera poses."""
        return self.pose_v

    def update(self, ok):
        """Keep only the image where ok is 1.

        Args:
            ok: vector of integers in {0,1}.
        """
        self.fn_v = self.fn_v[ok==1]
        self.pose_v = self.pose_v[ok==1]
        self.size = self.fn_v.shape[0]
    
    def check_idx(self, idx):
        """Checks whether the image index idx is within bounds.

        Args:
            idx: integer index.
        """
        if idx >= self.size:
            raise ValueError("You ask for an image out of bounds."
                    "idx >= size: %d >= %d"%(idx, self.size))
    
    def get_img_fn(self, idx):
        """Returns the full path image filename with index idx.

        Args:
            idx: integer index.
        """
        self.check_idx(idx)
        return('%s/%s'%(self.img_dir, self.fn_v[idx]))

    def get_pose(self, idx):
        """Returns pose of the idx-th image."""
        self.check_idx(idx)
        return self.pose_v[idx]

    def proc_img(self, img):
        """Processes the image. The image may need to be resized or denoised.
        """       
        raise NotImplementedError('Must be implemented by subclasses')

    def get_img(self, idx):
        """Return the processed image at index idx.

        Args:
            idx: integer image index.
        """
        self.check_idx(idx)
        img_fn = '%s/%s'%(self.img_dir, self.fn_v[idx])
        #print("img_fn: %s"%img_fn)
        img = cv2.imread(img_fn)
        img = self.proc_img(img)
        return img

    def get_semantic_img(self, idx):
        """Returns the processed semantic map for the image at index idx."""
        raise NotImplementedError('Must be implemented by subclasses')


class CMUSurvey(Survey):
    """Survey implementation for the CMU-Seasons dataset."""
    def __init__(self, meta_fn, img_dir, seg_dir):
        """Constructs a CMU-Seasons survey.
        
        Args: See base class.
        """
        super(CMUSurvey, self).__init__(meta_fn, img_dir, seg_dir)

    def proc_img(self, img, pixel_border=1):
        """Processes the image.

        Removes the border noisy segmentation and resize the image by 0.5.
        """
        img = img[pixel_border:-pixel_border, pixel_border:-pixel_border]
        img = cv2.resize(img, None, fx=0.5, fy=0.5,
                interpolation=cv2.INTER_NEAREST)
        return img

    def get_edge_img(self, idx):
        """Returns the resized edge image for index idx.

        Args:
            idx: integer index.
        """
        self.check_idx(idx)
        edge_fn = 'data/se/%s'%self.fn_v[idx]
        edge_img = cv2.imread(edge_fn, cv2.IMREAD_UNCHANGED)
        edge_img = cv2.resize(edge_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        return edge_img

    def get_semantic_img(self, idx, pixel_border=1):
        """Processes and returns the label semantic map for the image at index idx.

        Denoise the label segmentation image and resize it by 0.5.
        """
        self.check_idx(idx)

        # as in icra submission
        #sem_fn = '%s_clean/%s.png'%(os.path.dirname(self.seg_dir),
        #        self.fn_v[idx].split(".")[0])
        #if not os.path.exists(sem_fn):
        #    sem_fn = '%s/%s.png'%(self.seg_dir, self.fn_v[idx].split(".")[0])
        #    if not os.path.exists(sem_fn):
        #        raise ValueError("No such file: %s"%sem_fn)
        
        ## TODO: for better perfs
        sem_fn = '%s/%s.png'%(self.seg_dir, self.fn_v[idx].split(".")[0])
        if not os.path.exists(sem_fn):
            raise ValueError("No such file: %s"%sem_fn)

        #print("%d\t%s"%(idx, sem_fn))
        #if idx > 55:
        #    input('wait')

        sem_img = cv2.imread(sem_fn, cv2.IMREAD_UNCHANGED)
        if sem_img.ndim == 2:
            sem_img = semantic_proc.lab2col(sem_img)
        
        # TODO: clean this
        if sem_img.shape[1] == 1024:
            sem_img = self.proc_img(sem_img)
        elif sem_img.shape[1] == 1022: # I already took the border pixel out
            sem_img = cv2.resize(sem_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        elif sem_img.shape[1] == 512:
            sem_img = sem_img[pixel_border:-pixel_border, pixel_border:-pixel_border]
        return sem_img

 
class SymphonySurvey(Survey):
    """Survey implementation for the Symphony dataset.
    """
    def __init__(self, meta_fn, img_dir, seg_dir, mask_dir):
        """Constructs a Symphony survey.
        
        Args: See base class.
            mask_dir: directory with hand-made water segmentation.
        """
        super(SymphonySurvey, self).__init__(meta_fn, img_dir, seg_dir)
        self.mask_dir = mask_dir

    def proc_img(self, img, w=700):
        """Processes the image.

        Crops the image width, by default to 700.
        """
        img = img[:,:w]
        return img

    def get_mask(self, idx):
        """Returns the water semantic segmentation for idx-th image.

        Returns an image of shape (h,w) with h and w the height and width of
        the idx-th image. Image pixels are 255 where there is water.
        """
        self.check_idx(idx)
        img_fn = '%s/%s.png'%(self.mask_dir, self.fn_v[idx].split(".")[0])
        img = cv2.imread(img_fn, cv2.IMREAD_UNCHANGED)
        return img


    def get_semantic_img(self, idx):
        """Processes and returns the label semantic map for the idx-th image.

        Denoise the label segmentation image. The water and vegetation are
        especially noisy since the segmentation network is not finetuned to the
        lake dataset.
        """
        self.check_idx(idx)
        sem_fn = '%s/%s.png'%(self.seg_dir, self.fn_v[idx].split(".")[0])
        #print('sem_fn: %s'%sem_fn)
        sem_img = cv2.imread(sem_fn, cv2.IMREAD_UNCHANGED)
        sem_img = self.proc_img(sem_img)
        lab_img = tools_sem.col2lab(sem_img) # color map -> label map
        
        # vegetation denoising
        lab_img_new = lab_img.copy()
        lab_img_new[lab_img==4] = 8
        lab_img_new[lab_img==3] = 8
        
        # water denoising
        mask_img = self.get_mask(idx) # get hand-made water segmentation
        lab_img_new[mask_img==255] = 255
        lab_img_new[lab_img_new==0] = 8
        lab_img_new[lab_img_new==255] = 0
        sem_img_new = tools_sem.lab2col(lab_img_new)
        return sem_img_new


class SurveyFactory(object):
    """Factory class for surveys.
    
    Taken from 
    https://realpython.com/factory-method-python/#separate-object-creation-to-provide-common-interface
    """
    def __init__(self):
        self._builders = {}
        self._builders["cmu"] = CMUSurvey
        self._builders["symphony"] = SymphonySurvey

    def register_builder(self, key, builder):
        self._builders[key] = builder

    def create(self, key, **kwargs):
        builder = self._builders[key]
        if not builder:
            raise ValueError(key)
        return builder(**kwargs)


if __name__=='__main__':

    data = "lake"
    # CMU
    if data == "cmu":
        meta_dir = "meta/cmu/surveys"
        img_dir = "%s/datasets/Extended-CMU-Seasons/"%cst.WS_DIR
        seg_dir = "%s/tf/cross-season-segmentation/res/ext_cmu/"%cst.WS_DIR 
        slice_id, cam_id, survey_id = 24, 0, 2
        
        if survey_id == -1:
            meta_fn = "%s/%d/c%d_db.txt"%(meta_dir, slice_id, cam_id)
        else:
            meta_fn = "%s/%d/c%d_%d.txt"%(meta_dir, slice_id, cam_id, survey_id)
        #survey = CMUSurvey(meta_fn, img_dir, seg_dir)
        surveyFactory = SurveyFactory()
        kwargs = {"meta_fn": meta_fn, "img_dir": img_dir, "seg_dir": seg_dir}
        survey = surveyFactory.create("cmu", **kwargs)
        
        for idx in range(survey.size):
            img = survey.get_img(idx)
            cv2.imshow("img", img)
            if (cv2.waitKey(0) & 0xFF) == ord("q"):
                exit(0)

    elif data == "lake":
        meta_dir = "meta/lake/surveys"
        img_dir = "/mnt/lake/VBags/"
        seg_dir = "%s/tf/cross-season-segmentation/res/icra_retrieval/"%cst.WS_DIR
        mask_dir = "%s/datasets/lake/datasets/icra_retrieval/water/global"%cst.WS_DIR
        slice_id = 0 # always
        cam_id = 0 # always
        survey_id = 0
 
        if survey_id == -1:
            meta_fn = "%s/%d/c%d_db.txt"%(meta_dir, slice_id, cam_id)
        else:
            meta_fn = "%s/%d/c%d_%d.txt"%(meta_dir, slice_id, cam_id, survey_id)       
        surveyFactory = SurveyFactory()
        kwargs = {"meta_fn": meta_fn, "img_dir": img_dir, "seg_dir": seg_dir,
                "mask_dir": mask_dir}
        survey = surveyFactory.create("symphony", **kwargs)
        for idx in range(survey.size):
            print(survey.get_img_fn(idx))

    else:
        raise ValueError("I don't know this dataset: %s"%data)
