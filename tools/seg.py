""" """
import os, glob
import time

import numpy as np

import torch
import torchvision.transforms as standard_transforms


import third_party.cross-seasons-segmentation as css

from css.models import pspnet
from css.models import model_configs
from css.utils.segmentor_cmu import Segmentor
from css.datasets import cityscapes
from css.utils.misc import rename_keys_to_match
import css.utils.joint_transforms as joint_transforms

# colmap output dir
MACHINE = 1
if MACHINE == 0:
    DATASET_DIR = '/home/abenbihi/ws/datasets/'
    WS_DIR = '/home/abenbihi/ws/'
    EXT_IMG_DIR = '/mnt/data_drive/dataset/Extended-CMU-Seasons/'
elif MACHINE == 1:
    DATASET_DIR = '/home/gpu_user/assia/ws/datasets/'
    WS_DIR = '/home/gpu_user/assia/ws/'
    EXT_IMG_DIR = '/home/gpu_user/assia/ws/datasets/Extended-CMU-Seasons/'
    #EXT_IMG_DIR = '/mnt/dataX/assia/Extended-CMU-Seasons/'
    #DATA_DIR = '/home/abenbihi/ws/datasets/CMU-Seasons/'
else:
    print('Get you MTF MACHINE macro correct !')
    exit(1)

    
NETWORK_FILE = 'meta/weights/css/CMU-CS-Vistas-CE.pth'
NUM_CLASS = 19


def run_net(filenames_ims, filenames_segs):
    # network model
    print("Using CUDA" if torch.cuda.is_available() else "Using CPU")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Network and weight loading
    model_config = model_configs.PspnetCityscapesConfig()
    net = pspnet.PSPNet().to(device)
    print('load model ' + NETWORK_FILE)
    state_dict = torch.load(NETWORK_FILE, map_location=lambda storage, 
            loc: storage)
    # needed since we slightly changed the structure of the network in pspnet
    state_dict = rename_keys_to_match(state_dict)
    net.load_state_dict(state_dict)
    net.eval()


    # data proc
    input_transform = model_config.input_transform
    pre_validation_transform = model_config.pre_validation_transform
    # make sure crop size and stride same as during training
    sliding_crop = joint_transforms.SlidingCropImageOnly(
        713, 2/3.)


    # encapsulate pytorch model in Segmentor class
    print("Class number: %d"%net.n_classes) # 19
    segmentor = Segmentor( net, net.n_classes, colorize_fcn =
            cityscapes.colorize_mask, n_slices_per_pass = 10)

    # let's go
    count = 1
    t0 = time.time()
    for i, im_file in enumerate(filenames_ims):
        save_path = filenames_segs[i]
        tnow = time.time()
        print( "[%d/%d (%.1fs/%.1fs)] %s" % (count, len(filenames_ims), 
            tnow - t0, (tnow - t0) / count * len(filenames_ims), im_file))
        #print(save_path)

        segmentor.run_and_save( im_file, save_path, '',
                pre_sliding_crop_transform = pre_validation_transform,
                sliding_crop = sliding_crop, input_transform = input_transform,
                skip_if_seg_exists = True, use_gpu = True, save_logits=False)
        count += 1 


def run_net_wo_resize(filenames_ims, filenames_segs):
    # network model
    print("Using CUDA" if torch.cuda.is_available() else "Using CPU")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Network and weight loading
    model_config = model_configs.PspnetCityscapesConfig()
    net = model_config.init_network().to(device)
    #net = pspnet.PSPNet().to(device)
    print('load model ' + NETWORK_FILE)
    state_dict = torch.load(NETWORK_FILE, map_location=lambda storage, 
            loc: storage)
    # needed since we slightly changed the structure of the network in pspnet
    state_dict = rename_keys_to_match(state_dict)
    net.load_state_dict(state_dict)
    net.eval()

    # data proc
    input_transform = model_config.input_transform
    #pre_validation_transform = model_config.pre_validation_transform
    # make sure crop size and stride same as during training
    sliding_crop = joint_transforms.SlidingCropImageOnly(
        713, 2/3.)

    # encapsulate pytorch model in Segmentor class
    print("Class number: %d"%net.n_classes) # 19
    segmentor = Segmentor( net, net.n_classes, colorize_fcn =
            cityscapes.colorize_mask, n_slices_per_pass = 10)

    # let's go
    count = 1
    t0 = time.time()
    for i, im_file in enumerate(filenames_ims):
        save_path = filenames_segs[i]
        tnow = time.time()
        print( "[%d/%d (%.1fs/%.1fs)] %s" % (count, len(filenames_ims), 
            tnow - t0, (tnow - t0) / count * len(filenames_ims), im_file))
        #print(save_path)

        segmentor.run_and_save( im_file, save_path, '',
                pre_sliding_crop_transform = None,
                sliding_crop = sliding_crop, input_transform = input_transform,
                skip_if_seg_exists = True, use_gpu = True, save_logits=False)
        count += 1 


def segment_slice(slice_id):
    # output dir
    save_folder = 'res/ext_cmu/slice%d/'%(slice_id)
    if not os.path.exists('%s/query/'%save_folder):
        os.makedirs('%s/query/'%save_folder)
    if not os.path.exists('%s/database/'%save_folder):
        os.makedirs('%s/database/'%save_folder)
    
    q_fn = glob.glob('%s/slice%d/query/*'%(EXT_IMG_DIR, slice_id))
    db_fn = glob.glob('%s/slice%d/database/*'%(EXT_IMG_DIR, slice_id))

    filenames_ims = q_fn + db_fn
    print(filenames_ims[0])
    print(filenames_ims[-1])

    filenames_segs = ['%s/%s.png' %(save_folder, 
        ('/'.join(l.split("/")[-2:]).split(".")[0])) 
        for l in filenames_ims]
    print(filenames_segs[0])
    print(filenames_segs[-1])
    
    #run_net(filenames_ims, filenames_segs)
    run_net_wo_resize(filenames_ims, filenames_segs)


def segment_traversal(args):
    # output dir
    
    if args.survey_id == -1:
        meta_fn = "meta/cmu/survey/%d/c%d_db.txt"%(args.slice_id, args.cam_id)
        save_folder = 'res/seg/slice%d/database/'%(args.slice_id)
    else:
        meta_fn = "meta/cmu/survey/%d/c%d_%d.txt"%(args.slice_id, args.cam_id, args.survey_id)
        save_folder = 'res/seg/slice%d/query/'%(args.slice_id)
    
    img_fn = np.loadtxt(meta_fn, dtype=str)[:,0]

    filenames_segs = ['%s/%s.png' %(save_folder, 
        ('/'.join(l.split("/")[-2:]).split(".")[0])) 
        for l in filenames_ims]
    print(filenames_segs[0])
    print(filenames_segs[-1])
    
    #run_net(filenames_ims, filenames_segs)
    #run_net_wo_resize(filenames_ims, filenames_segs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str)
    parser.add_argument("--slice_id", type=int)
    parser.add_argument("--cam_id", type=int)
    parser.add_argument("--survey_id", type=int)
    args = parse.parse_args()
 
    # output dir
    save_folder = 'res/seg/slice%d/'%(args.slice_id)
    if not os.path.exists('%s/query/'%save_folder):
        os.makedirs('%s/query/'%save_folder)
    if not os.path.exists('%s/database/'%save_folder):
        os.makedirs('%s/database/'%save_folder)
   
    segment_traversal(args)

    #for slice_id in [7,6,8]:
    #    #segment_slice(slice_id)
    #    segment_slice(slice_id)
