"""Various constants."""
import numpy as np

MACHINE = 2
DATA='inria_holidays'

if MACHINE == -1:
    WS_DIR = '/home/ws/' # docker
elif MACHINE == 0:
    WS_DIR = '/home/abenbihi/ws/'
elif MACHINE == 2:
    WS_DIR = '/home/gpu_user/assia/ws/' # gpu
else:
    print('Get you mother fucking MACHINE macro correct in cst.py')
    exit(1)

if DATA=='inria_holidays':
    DATA_DIR = '%s/datasets/inria_holidays/jpg/'%WS_DIR
elif DATA=='aachen':
    print("Not there yet. SOON")
    exit(1)
else:
    print('Error: cst.py: unknown dataset: %s. Set DATA correctly in tools/cst.py.'%DATA)
    exit(1)

DATASET_DIR = '%s/datasets/'%WS_DIR
SEG_ROOT_DIR = '%s/tf/cross-season-segmentation/res/'%WS_DIR
NUM_CLASS = 19
PIXEL_BORDER = 1 # border pixels on which the segmentation is fucked up
SKY_LAB_ID = 10

## CMU-Seasons
#MACHINE = 0
#if MACHINE == 0:
#    #EXT_IMG_DIR = '/mnt/data_drive/dataset/Extended-CMU-Seasons/'
#    EXT_IMG_DIR = '%s/datasets/Extended-CMU-Seasons/'%WS_DIR
#    #DATA_DIR = '/mnt/data_drive/dataset/CMU-Seasons/'
#elif MACHINE == 1:
#    EXT_IMG_DIR = '%s/datasets/Extended-CMU-Seasons/'%WS_DIR
#    #DATA_DIR = '/home/abenbihi/ws/datasets/CMU-Seasons/'
#else:
#    print('Get you MTF MACHINE macro correct !')
#    exit(1)

EXT_IMG_DIR = '/mnt/data_drive/dataset/Extended-CMU-Seasons/'
EXT_IMG_DIR = '%s/datasets/Extended-CMU-Seasons/'%WS_DIR
SEG_DIR     = '%s/ext_cmu/'%SEG_ROOT_DIR
SEG_CLEAN_DIR     = '%s/ext_cmu_clean/'%SEG_ROOT_DIR
META_DIR = '%s/life_saver/datasets/CMU-Seasons/meta/'%WS_DIR


# Lake 
LAKE_EXT_IMG_DIR = '/mnt/lake/VBags/'
LAKE_SEG_DIR  = '%s/icra_retrieval/'%SEG_ROOT_DIR
LAKE_META_DIR = '%s/datasets/lake/lake/meta/retrieval/'%WS_DIR
LAKE_MASK_DIR = '%s/datasets/lake/datasets/icra_retrieval/water/global/'%WS_DIR


# cityscapes labels and colors
LABEL_NUM = 19
label_name = [
    'road',
    'sidewalk',
    'building',
    'wall',
    'fence',
    'pole',
    'traffic light',
    'traffic sign',
    'vegetation',
    'terrain',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle', 
    'mask', # added by Assia
]

# rgb
palette = [[128, 64, 128], 
        [244, 35, 232], 
        [70, 70, 70], 
        [102, 102, 156], 
        [190, 153, 153], 
        [153, 153, 153], 
        [250, 170, 30],
        [220, 220, 0], 
        [107, 142, 35], 
        [152, 251, 152], 
        [70, 130, 180], 
        [220, 20, 60], 
        [255, 0, 0], 
        [0, 0, 142], 
        [0, 0, 70],
        [0, 60, 100], 
        [0, 80, 100], 
        [0, 0, 230], 
        [119, 11, 32],
        [0, 0, 0]] # class 20: ignored, I added it, not cityscapes

palette_bgr = [ [l[2], l[1], l[0]] for l in palette]

label_ignore = 19 # mask that I artificially introduce
# all moving objects in cityscapes
labels_foreground = np.array([ 11, 12, 13, 14, 15, 16, 17, 18]) 
