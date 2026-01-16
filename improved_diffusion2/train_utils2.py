import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from .discriminators import ImagePool
import cv2

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

def DiceLoss(y_true, y_pred):
    ndims = len(list(y_pred.size())) - 2
    vol_axes = list(range(2, ndims + 2))
    top = 2 * (y_true * y_pred).sum(dim=vol_axes)
    bottom = th.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
    dice = th.mean(top / bottom)
    return 1.0 - dice

def Baseline2DLoss(y_true, y_pred):
    dice_loss = DiceLoss(y_true, y_pred)
    criterion = th.nn.BCELoss()
    cross_loss = criterion(y_pred, y_true)
    return dice_loss + cross_loss

class TrainLoop:
    def __init__(
        self,
        *,
        model_A,
        diffusion_A,
        model_B,
        diffusion_B,
        model,
        diffusion,
        netD_s,
        seg_model,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model_A = model_A
        self.model_B = model_B
        self.diffusion_A = diffusion_A
        self.diffusion_B = diffusion_B
        
        self.model = model
        self.diffusion = diffusion
        self.netD_s = netD_s
        self.seg_model = seg_model
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        self.fake_img_pool = ImagePool(50)

        # self.model_params = list(self.model.parameters())
        self.model_params = [param for param in self.model.parameters() if param.requires_grad]
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        self.opt_D = AdamW(list(self.netD_s.parameters()), lr=self.lr * 10, weight_decay=self.weight_decay)
        self.opt_S = AdamW(list(self.seg_model.parameters()), lr=self.lr * 10, weight_decay=self.weight_decay)
        
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch_data = next(self.data)
            self.run_step(batch_data)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch_data):
        self.forward_backward(batch_data)
        # if self.use_fp16:
        #     self.optimize_fp16()
        # else:
        #     self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch_data):
        batch = batch_data['data_A'].float()
        vessel_img = batch_data['vessel_img_A'].float()
        thick_label = batch_data['thick_label_A'].float()
        data_B = batch_data['data_B'].float()
        
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_data_B = data_B[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {}
            micro_vessel_img = vessel_img[i : i + self.microbatch].to(dist_util.dev())
            micro_thick_label = thick_label[i : i + self.microbatch].to(dist_util.dev())
            last_batch = (i + self.microbatch) >= batch.shape[0]
            
            out_x = micro.clone()
            backward_indices = list(range(30))[::-1][int(30*(1 - 0.3)):]
            for j in backward_indices:
                t_style = th.tensor([j]*micro.shape[0], device=micro.device)
                out_style = self.diffusion.ddim_sample(self.model,
                                                          out_x,
                                                          micro_vessel_img,
                                                          t_style,)
                out_x = out_style['sample']
            # import pdb;pdb.set_trace()
            if self.step % 500 == 0:
                predict_img = out_x[0][0].detach().cpu().numpy()
                predict_img = (predict_img + 1) * 127.5
                predict_img[predict_img < 0] = 0
                predict_img[predict_img > 255] = 255
                cv2.imwrite(os.path.join(logger.get_dir(), f"{str(self.step).zfill(6)}_recon.png"), predict_img)
            # import pdb;pdb.set_trace()
            loss_G = self.diffusion.loss_gan(self.netD_s(out_x), True)
            loss_G2 = Baseline2DLoss(self.seg_model(out_x), micro_vessel_img)
            loss_G = loss_G + loss_G2 * 0.1
            
            pred_real = self.netD_s(micro_data_B)
            loss_D_real = self.diffusion.loss_gan(pred_real, True)
            fake_img = self.fake_img_pool.query(out_x)
            pred_fake = self.netD_s(fake_img.detach())
            loss_D_fake = self.diffusion.loss_gan(pred_fake, False)
            loss_S = Baseline2DLoss(self.seg_model(out_x.detach()), micro_vessel_img)
            
            pred_real_ = th.mean(pred_real, [1, 2, 3])
            pred_fake_ = th.mean(pred_fake, [1, 2, 3])
            acc_num = (pred_real_ > 0.5).sum() + (pred_fake_ < 0.5).sum()
            acc = acc_num / (pred_real.shape[0] * 2)
            print(acc)
            
            if self.step > 100 and float(acc) > 0.75:
                if self.use_fp16:
                    loss_scale = 2 ** self.lg_loss_scale
                    (loss_G * loss_scale).backward()
                else:
                    loss_G.backward()
                
                if self.use_fp16:
                    self.optimize_fp16()
                else:
                    self.optimize_normal()
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            print(loss_D)
            self.opt_D.zero_grad()
            loss_D.backward()
            self._anneal_lr2()
            self.opt_D.step()
            
    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
    
    def _anneal_lr2(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done) * 10
        for param_group in self.opt_D.param_groups:
            param_group["lr"] = lr
        for param_group in self.opt_S.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            trainable_params = [param for param in self.model.parameters() if param.requires_grad]
            master_params = unflatten_master_params(trainable_params, master_params)
            # master_params = unflatten_master_params(
            #     self.model.parameters(), master_params
            # )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate((name, param) for name, param in self.model.named_parameters() if param.requires_grad):
            state_dict[name] = master_params[i]
        # for i, (name, _value) in enumerate(self.model.named_parameters()):
        #     assert name in state_dict
        #     state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
