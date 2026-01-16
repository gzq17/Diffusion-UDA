import argparse
import sys
sys.path.append('./')
from improved_diffusion2.segmentation_model import SegmentationModel2D
import os
import datetime
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import blobfile as bf
import cv2
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as F
import random
seed = 0# ROSE:0->71.36;72.61;74.13
#new: 0 3w:73.53,74.45   0 10w:73.31 74.26  0 12w: 73.63 74.50  12w 250
#OCTA: 0:74.92 250:74.46, 
np.random.seed(seed)
random.seed(seed)
torch.random.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=int, default=1)
parser.add_argument('--generated_path', type=str, default="")
parser.add_argument('--data_path', type=str, default="") 
parser.add_argument('--batch_size', type=int, default=64, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--iter_num', type=int, default=1)
parser.add_argument('--seg_lr', type=float, default=1e-3)
parser.add_argument('--load_name', type=str, default="")
parser.add_argument('--num_fea_list', type=float, default=[1, 16, 32, 64, 128, 256])
args = parser.parse_args()

device = 'cuda'

def DiceLoss(y_true, y_pred):
    ndims = len(list(y_pred.size())) - 2
    vol_axes = list(range(2, ndims + 2))
    top = 2 * (y_true * y_pred).sum(dim=vol_axes)
    bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
    dice = torch.mean(top / bottom)
    return 1.0 - dice

def Baseline2DLoss(y_true, y_pred):
    dice_loss = DiceLoss(y_true, y_pred)
    criterion = nn.BCELoss()
    cross_loss = criterion(y_pred, y_true)
    return dice_loss, cross_loss

def adjust_lr(optimizer, epoch, lr):
    lr_c = lr * ((1 - epoch/(args.epochs + 1)) ** 0.9)
    for p in optimizer.param_groups:
        p['lr'] = lr_c

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def train(train_loader, val_name_list):
    model_dir = '/data/guozhanqiang/binary2gray/Improve-DDPM/' + datetime.datetime.now().strftime("segment-%Y-%m-%d-%H-%M-%S-%f") + '/'
    os.makedirs(model_dir)
    loss_file_name = model_dir + 'loss.txt'
    loss_file = open(loss_file_name, 'w')
    loss_file.write(args.data_path + '\n')
    loss_file.flush()
    
    seg_model = SegmentationModel2D(1, 1, activate='relu').to(device)
    lr = 1e-4
    print(get_parameter_number(seg_model))
    # import pdb;pdb.set_trace()
    seg_model.train()
    seg_optimizer = torch.optim.Adam(seg_model.parameters(), lr=lr)
    loss_name = ['dice_loss', 'cross_loss', 'loss_sum']
    val_dice_max = 0.0
    epoch_max_index = 0
    test_best_dice = 0.0
    for epoch in range(0, args.epochs):
        seg_model.train()
        loss_list_sum = [0, 0, 0]
        if epoch % 10 == 0:
            adjust_lr(seg_optimizer, epoch, args.seg_lr)
        for i, data in enumerate(train_loader):
            img2d = data['data'].to(device).float()
            lbl2d = data['vessel_img'].to(device).float()
            result2d = seg_model(img2d)
            dice_loss, cross_loss = Baseline2DLoss(lbl2d, result2d)
            loss_sum = cross_loss + dice_loss
            seg_optimizer.zero_grad()
            loss_sum.backward()
            seg_optimizer.step()
            loss_list_one = [float(dice_loss), float(cross_loss), float(loss_sum)]
            for ii in range(0, len(loss_name)):
                loss_list_sum[ii] = loss_list_sum[ii] + loss_list_one[ii]
            if i % 5 == 0:
                info = 'epoch:{}/{}, {}, '.format(epoch, args.epochs, i)
                for ii in range(0, len(loss_name)):
                    info += loss_name[ii] + ':{:.4f}  '.format(loss_list_one[ii])
                print(info)
                loss_file.write(info + '\n')
                loss_file.flush()
                val_info, val_dice = test_model(seg_model, val_name_list, epoch, model_dir)
                if val_dice > test_best_dice:
                    torch.save(seg_model.state_dict(), model_dir + 'best2.pth')
                test_best_dice = max(test_best_dice, val_dice)
                
        val_info, val_dice = test_model(seg_model, val_name_list, epoch, model_dir)
        info_epoch = 'epoch:{}/{} '.format(epoch, args.epochs)
        for ii in range(0, len(loss_name)):
            info_epoch += loss_name[ii] + ':{:.4f}  '.format(loss_list_sum[ii] / (i + 1))
        print(info_epoch + '\n')
        if val_dice > val_dice_max:
            val_dice_max = val_dice
            epoch_max_index = epoch
            torch.save(seg_model.state_dict(), model_dir + 'best.pth')
        loss_file.write(info_epoch + '\n')
        loss_file.write(val_info + '\n')
        loss_file.write('epoch_max_index:{}, val_dice_max:{:.2f}\n'.format(epoch_max_index, val_dice_max))
        loss_file.flush()
        print(epoch_max_index, val_dice_max, test_best_dice)
        if (epoch + 1) % 200 == 0:
            torch.save(seg_model.state_dict(), model_dir + str(epoch + 1) +'.pth')
        else:
            torch.save(seg_model.state_dict(), model_dir + 'last.pth')

def test_model(model, name_list, epoch, model_dir, save=False):
    model.eval()
    dice_sum = 0
    if save:
        result_path = args.data_path + f"Temp_result_{args.iter_num}/"
        os.makedirs(result_path, exist_ok=True)
    for name in name_list:
        img_path = args.data_path + 'Original/' + name + '.png'
        lbl_path = args.data_path + 'Ground_truth/' + name + '_label.png'
        arr = cv2.imread(img_path, 0)
        lbl = cv2.imread(lbl_path, 0)
        lbl[lbl == 255] = 1
        arr = arr.astype(np.float32) / 127.5 - 1
        xx_list, yy_list = [], []
        sz = 256
        over_sz = 128
        for ii in range(0, arr.shape[0], over_sz):
            if ii + sz > arr.shape[0]:
                break
            xx_list.append(ii)
        if arr.shape[0] - sz not in xx_list:
            xx_list.append(arr.shape[0] - sz)
        for ii in range(0, arr.shape[1], over_sz):
            if ii + sz > arr.shape[1]:
                break
            yy_list.append(ii)
        if arr.shape[1] - sz not in yy_list:
            yy_list.append(arr.shape[1] - sz)
        arr = arr[np.newaxis, :, :]
        img_list = []
        for xx in xx_list:
            for yy in yy_list:
                img_list.append(arr[np.newaxis, :, xx:xx+sz, yy:yy+sz])
        img = np.concatenate(img_list, axis=0)
        img = torch.from_numpy(img).to(device).float()
        result_list = model(img)
        result_list = result_list.squeeze().detach().cpu().numpy()
        ii = 0
        result_img = np.zeros(lbl.shape)
        count_img = np.zeros(lbl.shape)
        for xx in xx_list:
            for yy in yy_list:
                result_img[xx:xx+sz, yy:yy+sz] += result_list[ii]
                count_img[xx:xx+sz, yy:yy+sz] += 1.0
                ii += 1
        result_img = result_img / count_img
        result_img[result_img > 0.5] = 1
        result_img[result_img <= 0.5] = 0
        dice_one = 2.0 * (((result_img == 1) & (lbl == 1)).sum()) / ((lbl == 1).sum() + (result_img == 1).sum())
        dice_sum += dice_one
        if save:
            cv2.imwrite(result_path + f"{name}_label.png", result_img * 255.0)
    val_info = 'val info, epoch:{} '.format(epoch)
    val_info += 'val dice:{:.2f}'.format(float(dice_sum / len(name_list)) * 100)
    print(val_info + '\n')
    model.train()
    return val_info, float(dice_sum / len(name_list) * 100)

def test_model2(model, path_dir, epoch, save=False):
    name_list_ = sorted(os.listdir(path_dir))
    name_list = []
    if save:
        result_path = path_dir + 'result1/'
        os.makedirs(result_path, exist_ok=True)
    for name in name_list_:
        if '_label.png' in name:
            name_list.append(name[:-10])
    dice_sum = 0
    for name in name_list:
        img_path = path_dir + name + '.png'
        lbl_path = path_dir + name + '_label.png'
        arr = cv2.imread(img_path, 0)
        lbl = cv2.imread(lbl_path, 0)
        lbl[lbl == 255] = 1
        arr = arr.astype(np.float32) / 127.5 - 1
        
        img = torch.from_numpy(arr[np.newaxis, np.newaxis, :, :]).to(device).float()
        result = model(img)
        result = result.squeeze().detach().cpu().numpy()
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        if save:
            cv2.imwrite(result_path + f"{name}_seg_result.png", result * 255.0)
        dice_one = 2.0 * (((result == 1) & (lbl == 1)).sum()) / ((lbl == 1).sum() + (lbl == 1).sum())
        dice_sum += dice_one
        # print(dice_one)
    val_info = 'val info, epoch:{} '.format(epoch)
    val_info += 'val dice:{:.2f}'.format(float(dice_sum / len(name_list)) * 100)
    print(val_info + '\n')
    model.train()
    return val_info, float(dice_sum / len(name_list) * 100)

def read_txt(file_name=None):
    if file_name is None:
        return None
    name_list = []
    f = open(file_name, 'r')
    a = f.readlines()
    for name in a:
        name_list.append(name[:-1])
    return name_list

def main():
    if args.train==1:
        train_set = RetainDatasetTarget2(args.generated_path, args.generated_path)
        print(train_set.__len__())
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_data_list = sorted(read_txt(f'{args.data_path}test.txt'))[:4]
        print(val_data_list)
        print('train set:', train_set.__len__(), 'val set:', len(val_data_list))
        train(train_loader, val_data_list)
    else:
        val_data_list = sorted(read_txt(f'{args.data_path}test.txt')) + sorted(read_txt(f'{args.data_path}train.txt'))
        print(len(val_data_list))
        seg_model = SegmentationModel2D(1, 1, activate='relu').to(device)
        seg_model.load_state_dict(torch.load(args.load_name))
        test_model(seg_model, val_data_list, 0, args.load_name, save=True)

class RetainDatasetTarget2(Dataset):
    def __init__(self, generated_path, generated_lbl_path):
        self.sz = 256
        
        self.generated_path = generated_path
        self.generated_lbl_path = generated_lbl_path
        self.name_list = []
        name_list_ = sorted(os.listdir(generated_path))
        for name in name_list_:
            if '_recon.png' in name:
                self.name_list.append(name[:-10])
    
    def __len__(self,):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        name = self.name_list[idx]
        img_path = self.generated_path + name + '_recon.png'
        lbl_path = self.generated_lbl_path + name + '_label.png'
        if not os.path.exists(lbl_path):
            lbl_path = self.generated_lbl_path + name[6:] + '_label.png'
        arr = cv2.imread(img_path, 0)
        lbl = cv2.imread(lbl_path, 0)
        lbl[lbl == 255] = 1
        arr = arr.astype(np.float32) / 255.0
        arr = arr[np.newaxis, :, :]
        arr = arr * 2.0 - 1.0
        return {'data': arr, 'vessel_img':torch.from_numpy(lbl[np.newaxis, :, :]), 'name': name}

if __name__ == '__main__':
    main()
