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
from improved_diffusion2.image_datasets import load_data, load_data3, load_data_target

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


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    logger.log(args.model_path)
    T_train_for = 50#50
    args.timestep_respacing = f"ddim{T_train_for}"
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")

    data = load_data3(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        vessel_label=args.vessel_label,
        label_path=args.label_path,
    )
    
    num = 0
    for ii in range(0, 2048 // args.batch_size):
        batch_data = next(data)
        batch = batch_data['data'].float().to(dist_util.dev())
        batch = batch * (-1.0)
        vessel_img = batch_data['vessel_img'].float().to(dist_util.dev())
        thick_vessel_img = batch_data['thick_label'].float().to(dist_util.dev())
        name_list = batch_data['name']
        print(batch.shape, vessel_img.shape)
        forward_indices = list(range(T_train_for))[:int(T_train_for*args.t0_ratio)]
        out_x = batch.clone()
        # import pdb;pdb.set_trace()
        with th.no_grad():
            for j in forward_indices:
                t_style = th.tensor([j]*batch.shape[0], device=batch.device)
                out_style = diffusion.ddim_reverse_sample(model,
                                                          out_x,
                                                          vessel_img,
                                                          t_style,)
                out_x = out_style['sample']
        for kk in range(0, len(name_list)):
            # import pdb;pdb.set_trace()
            predict_img = out_x[kk].detach().cpu().numpy()
            gt_img = batch[kk].detach().cpu().numpy()
            vessel = vessel_img[kk][0].detach().cpu().numpy()
            thick_vessel = thick_vessel_img[kk][0].detach().cpu().numpy()
            predict_img = (predict_img + 1) * 127.5
            gt_img = (gt_img + 1) * 127.5
            vessel = vessel * 255.0
            thick_vessel = thick_vessel * 255.0
            predict_img[predict_img < 0] = 0
            predict_img[predict_img > 255] = 255
            # predict_img = 255 - predict_img
            cv2.imwrite(os.path.join(logger.get_dir(), f"{str(num).zfill(5)}-{name_list[kk]}_t0.png"), predict_img[0])
            cv2.imwrite(os.path.join(logger.get_dir(), f"{str(num).zfill(5)}-{name_list[kk]}_label.png"), vessel)
            cv2.imwrite(os.path.join(logger.get_dir(), f"{str(num).zfill(5)}-{name_list[kk]}_thick_label.png"), thick_vessel)
            cv2.imwrite(os.path.join(logger.get_dir(), f"{str(num).zfill(5)}-{name_list[kk]}.png"), gt_img[0])
            num += 1

def create_argparser():
    defaults = dict(
        data_dir="",#post_data  OCTA-500
        vessel_label=True,
        label_path="Ground_truth/",
        clip_denoised=True,
        num_samples=36,
        batch_size=16,
        use_ddim=True,
        t0_ratio = 0.3,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
