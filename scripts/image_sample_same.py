"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import sys
sys.path.append('./')
import numpy as np
import torch as th
import torch.distributed as dist
from improved_diffusion2.image_datasets import load_data, load_data3

from improved_diffusion2 import dist_util, logger
import SimpleITK as sitk
import cv2
from PIL import Image
from improved_diffusion2.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import random
from torch.utils.data import DataLoader, Dataset
seed = 42
np.random.seed(seed)
random.seed(seed)
th.random.manual_seed(seed)

def load_data_target(data_dir_A, batch_size, deterministic=False):    
    name_list = sorted(os.listdir(data_dir_A))
    data_list_all_A = []
    for name in name_list:
        if '_thick_label.png' not in name:
            continue
        data_list_all_A.append(name[:-16])
    
    dataset = RetainDatasetTarget2(
        data_dir_A,
        data_list_all_A,
        choose_part=True
    )
    print(dataset.__len__())
    if deterministic:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    while True:
        yield from loader

class RetainDatasetTarget2(Dataset):
    def __init__(self, data_path_A, data_list_all_A, choose_part=False):
        self.data_path_A = data_path_A
        self.data_list_A = data_list_all_A
        self.sz = 256
        self.choose_part = choose_part
    
    def __len__(self,):
        return len(self.data_list_A)

    def __getitem__(self, idx):
        name_A = self.data_list_A[idx]
        img_path = self.data_path_A + name_A + '.png'
        lbl_path = self.data_path_A + name_A + '_label.png'
        thick_label_path = self.data_path_A + name_A + '_thick_label.png'
        arr = cv2.imread(img_path, 0)
        thick_lbl = cv2.imread(thick_label_path, 0)
        thick_lbl[thick_lbl == 255] = 1
        lbl = cv2.imread(lbl_path, 0)
        lbl[lbl == 255] = 1
        arr = arr.astype(np.float32) / 127.5 - 1
        
        return {
            'data': arr[np.newaxis, :, :],
            'vessel_img': lbl[np.newaxis, :, :],
            'thick_label': thick_lbl[np.newaxis, :, :],
            'name': name_A
        }


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    logger.log(args.model_path)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")

    data = load_data_target(
        data_dir_A=args.data_dir,
        batch_size=args.batch_size,
    )
    
    num = 0
    for ii in range(0, 2048 // args.batch_size):
        batch_data = next(data)
        batch = batch_data['data'].float().to(dist_util.dev())
        vessel_img = batch_data['vessel_img'].float().to(dist_util.dev())
        
        name_list = batch_data['name']
        print(batch.shape, vessel_img.shape)
        print(name_list)
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        model_kwargs = {}
        # import pdb;pdb.set_trace()
        sample = sample_fn(
            model,
            batch.shape,
            vessel_img,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True,
        )
        for kk in range(0, len(name_list)):
            # import pdb;pdb.set_trace()
            predict_img = sample[kk].detach().cpu().numpy()
            gt_img = batch[kk].detach().cpu().numpy()
            vessel = vessel_img[kk][0].detach().cpu().numpy()
            predict_img = (predict_img + 1) * 127.5
            gt_img = (gt_img + 1) * 127.5
            vessel = vessel * 255.0
            predict_img[predict_img < 0] = 0
            predict_img[predict_img > 255] = 255
            cv2.imwrite(os.path.join(logger.get_dir(), f"{name_list[kk]}_recon.png"), predict_img[0])
            cv2.imwrite(os.path.join(logger.get_dir(), f"{name_list[kk]}_label.png"), vessel)
            cv2.imwrite(os.path.join(logger.get_dir(), f"{name_list[kk]}.png"), gt_img[0])
            num += 1

def create_argparser():
    defaults = dict(
        data_dir="",#post_data  OCTA-500
        clip_denoised=True,
        num_samples=36,
        batch_size=64,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
