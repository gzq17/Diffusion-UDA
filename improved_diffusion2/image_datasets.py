from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import copy
import cv2
import os
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt

np.random.seed(1234)

def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    # if not data_dir:
    #     raise ValueError("unspecified data directory")
    # all_files = _list_image_files_recursively(data_dir)
    # classes = None
    # if class_cond:
    #     # Assume classes are the first part of the filename,
    #     # before an underscore.
    #     class_names = [bf.basename(path).split("_")[0] for path in all_files]
    #     sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    #     classes = [sorted_classes[x] for x in class_names]
    # data_list_all = sorted(read_txt(f'{data_dir}train.txt'))
    data_list_all = sorted(read_txt(f'{data_dir}test.txt'))
    data_list_all = data_list_all * 2
    # dataset = ImageDataset(
    #     image_size,
    #     all_files,
    #     classes=classes,
    #     shard=MPI.COMM_WORLD.Get_rank(),
    #     num_shards=MPI.COMM_WORLD.Get_size(),
    # )
    dataset = CoronaryLabel(
        data_dir,
        data_list_all,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    
    print(dataset.__len__())
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
    while True:
        yield from loader

def load_data3(
    *, data_dir, batch_size, label_path="Ground_truth/", deterministic=False,choose_part=False,vessel_label=True,
):
    data_list_all = sorted(read_txt(f'{data_dir}train.txt'))
    data_list_all = data_list_all
    if len(data_list_all) < 50:
        data_list_all = data_list_all * 10
    dataset = RetainDataset(
        data_dir,
        data_list_all,
        label_path=label_path,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        choose_part=choose_part,
        vessel_label=vessel_label,
    )
    
    print(dataset.__len__())
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
    while True:
        yield from loader

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def read_txt(file_name=None):
    if file_name is None:
        return None
    name_list = []
    f = open(file_name, 'r')
    a = f.readlines()
    for name in a:
        name_list.append(name[:-1])
    return name_list

class CoronaryLabel2(Dataset):

    def __init__(self, data_path, image_paths, shard=0, num_shards=1):
        self.data_path = data_path
        self.data_list = image_paths[shard:][::num_shards]
        self.latent_path = data_path.replace('AllData/', 'VQGANmodel/') + 'model_test/latent_result2/'
        print(self.latent_path)
    
    def __len__(self,):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        name = self.data_list[idx]
        label = np.load(self.data_path + 'size_sz_lbl/' + name + '_label.npy', mmap_mode='r')
        thick_label = np.load(self.data_path + 'vessel_center/' + name + '_thick_label.npy', mmap_mode='r')

        data_name = self.latent_path + name + '_latent.npy'
        data = np.load(data_name, mmap_mode='r')
        
        feature_name = self.data_path + 'feature_arr/' + name + '.npy'
        fea_arr = np.load(feature_name, mmap_mode='r')

        feature_name2 = self.data_path + 'feature_arr/' + name + '_08.npy'
        fea_arr2 = np.load(feature_name2, mmap_mode='r')

        feature_name3 = self.data_path + 'feature_arr/' + name + '_04.npy'
        fea_arr3 = np.load(feature_name3, mmap_mode='r')

        feature_name4 = self.data_path + 'feature_arr/' + name + '_02.npy'
        fea_arr4 = np.load(feature_name4, mmap_mode='r')

        return {'data': torch.from_numpy(data), 'vessel_img': torch.from_numpy(label)[None, ...], 
                'thick_label': torch.from_numpy(thick_label)[None, ...], 'fea_arr': torch.from_numpy(fea_arr),
                'fea_arr2': torch.from_numpy(fea_arr2),  'fea_arr3': torch.from_numpy(fea_arr3), 
                'fea_arr4': torch.from_numpy(fea_arr4),  'name':name}

def get_bound(x_c, c_p, max_x):
    x_b = x_c - c_p // 2
    if x_b < 0:
        x_b = 0
    if x_b + c_p > max_x:
        x_b = max_x - c_p
    return x_b

class CoronaryLabel(Dataset):

    def __init__(self, data_path, image_paths, shard=0, num_shards=1):
        self.data_path = data_path
        self.data_list = image_paths[shard:][::num_shards]
    
    def __len__(self,):
        return len(self.data_list)
    
    def crop_patch(self, img, label, thick_label):
        # import pdb;pdb.set_trace()
        crop_size = (96, 96, 96)
        if np.random.random() > 0.5:
            xx_b = np.random.randint(0, img.shape[0] - crop_size[0])
            yy_b = np.random.randint(0, img.shape[1] - crop_size[1])
            zz_b = np.random.randint(0, img.shape[2] - crop_size[2])
        else:
            label_index = np.where(label == 1)
            rand_index = np.random.randint(0, label_index[0].shape[0])
            xx_c, yy_c, zz_c = label_index[0][rand_index], label_index[1][rand_index], label_index[2][rand_index]
            xx_b, yy_b, zz_b = get_bound(xx_c, crop_size[0], img.shape[0]), get_bound(yy_c, crop_size[1], img.shape[1]), get_bound(zz_c, crop_size[2], img.shape[2])
            
        img_patch = img[xx_b: xx_b + crop_size[0], yy_b: yy_b + crop_size[1], zz_b: zz_b + crop_size[2]]
        lbl_patch = label[xx_b: xx_b + crop_size[0], yy_b: yy_b + crop_size[1], zz_b: zz_b + crop_size[2]]
        thick_label_patch = thick_label[xx_b: xx_b + crop_size[0], yy_b: yy_b + crop_size[1], zz_b: zz_b + crop_size[2]]
        return img_patch, lbl_patch, thick_label_patch
    
    def __getitem__(self, idx):
        name = self.data_list[idx]
        if os.path.exists(self.data_path + 'size_sz_lbl/' + name + '_label.npy'):
            label = np.load(self.data_path + 'size_sz_lbl/' + name + '_label.npy', mmap_mode='r')#size_sz_img
            thick_label = np.load(self.data_path + 'vessel_center/' + name + '_thick_label.npy', mmap_mode='r')###Temp_truth4
            data = np.load(self.data_path + 'size_sz_img/' + name + '.npy', mmap_mode='r')
        else:
            label = sitk.GetArrayFromImage(sitk.ReadImage(self.data_path + 'size_sz_lbl/' + name + '_label.nii.gz'))
            thick_label = sitk.GetArrayFromImage(sitk.ReadImage(self.data_path + 'vessel_center/' + name + '_thick_label.nii.gz'))
            data = sitk.GetArrayFromImage(sitk.ReadImage(self.data_path + 'size_sz_img/' + name + '.nii.gz'))
        # print(data.shape)
        # import pdb;pdb.set_trace()
        data = (data - data.min()) / (data.max() - data.min()) * 2.0 - 1.0
        # print(data.max(), data.min())
        img_patch, lbl_patch, thick_label_patch = self.crop_patch(data, label, thick_label)
        # print(img_patch.shape, lbl_patch.shape)
        return {'data': torch.from_numpy(img_patch)[None, ...], 'vessel_img': torch.from_numpy(lbl_patch)[None, ...],
                'thick_label': torch.from_numpy(thick_label_patch)[None, ...], 'name': name}

class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict

class RetainDataset(Dataset):
    def __init__(self, data_path, image_paths, label_path, shard=0, num_shards=1, vessel_label=True, choose_part=False):
        self.data_path = data_path
        self.data_list = image_paths[shard:][::num_shards]
        self.sz = 256
        self.vessel_label = vessel_label
        self.choose_part = choose_part
        self.label_path = label_path
    
    def __len__(self,):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        name = self.data_list[idx]
        img_path = self.data_path + 'Original/' + name + '.png'
        lbl_path = self.data_path + self.label_path + name + '_label.png'##Temp_truth4 Ground_truth
        thick_label_path = self.data_path + self.label_path + name + '_thick_label.png'
        arr = cv2.imread(img_path, 0)

        thick_lbl = cv2.imread(thick_label_path, 0)
        thick_lbl[thick_lbl == 255] = 1

        lbl = cv2.imread(lbl_path, 0)
        lbl[lbl == 255] = 1
        if not self.vessel_label:
            lbl[lbl == 1] = 0
            thick_lbl[thick_lbl == 1] = 0
        if arr.shape[0] > self.sz:
            # import pdb;pdb.set_trace()
            if self.choose_part and self.vessel_label:
                left, right = lbl[:, :arr.shape[1] // 2].sum(), lbl[:, arr.shape[1] // 2:].sum()
                ch_x, ch_y = arr.shape[0] - self.sz - 1, arr.shape[1] - self.sz - 1
                xx_b = np.random.randint(0, ch_x)
                if left > right:
                    yy_b = np.random.randint(ch_y // 2, ch_y)
                else:
                    yy_b = np.random.randint(0, ch_y // 2)
            else:
                xx_b = np.random.randint(0, arr.shape[0] - self.sz - 1)
                yy_b = np.random.randint(0, arr.shape[1] - self.sz - 1)
            # arr = arr[xx_b: xx_b + self.sz, yy_b: yy_b + self.sz, :]
            arr = arr[xx_b: xx_b + self.sz, yy_b: yy_b + self.sz]
            lbl = lbl[xx_b: xx_b + self.sz, yy_b: yy_b + self.sz]
            thick_lbl = thick_lbl[xx_b: xx_b + self.sz, yy_b: yy_b + self.sz]
        arr = arr.astype(np.float32) / 127.5 - 1
        return {'data': arr[np.newaxis, :, :], 'vessel_img':lbl[np.newaxis, :, :], 'thick_label':thick_lbl[np.newaxis, :, :],  'name': name}
    
def load_data_target(
    *, data_dir_A, data_dir_B, batch_size, image_size, class_cond=False, deterministic=False
):
    data_list_all_A = sorted(read_txt(f'{data_dir_A}train.txt'))
    
    data_list_all_B = sorted(read_txt(f'{data_dir_B}train.txt'))
    has_label_B = data_list_all_B[:5]
    
    dataset = RetainDatasetTarget(
        data_dir_A,
        data_dir_B,
        data_list_all_A,
        data_list_all_B,
        has_label_B=has_label_B,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        choose_part=True
    )
    
    print(dataset.__len__())
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
    while True:
        yield from loader

def load_data_target2(data_dir_A, data_dir_B, batch_size, image_size, class_cond=False, deterministic=False, source_have_label=True):    
    name_list = sorted(os.listdir(data_dir_A))
    data_list_all_A = []
    for name in name_list:
        if '_t0.png' not in name:
            continue
        data_list_all_A.append(name[:-7])
    
    data_list_all_B = sorted(read_txt(f'{data_dir_B}train.txt'))
    has_label_B = data_list_all_B[:5]
    
    dataset = RetainDatasetTarget2(
        data_dir_A,
        data_dir_B,
        data_list_all_A,
        data_list_all_B,
        has_label_B=has_label_B,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        choose_part=True,
        source_have_label=source_have_label
    )
    print(dataset.__len__())
    if deterministic:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    while True:
        yield from loader
    

class RetainDatasetTarget(Dataset):
    def __init__(self, data_path_A, data_path_B, data_list_all_A, data_list_all_B, has_label_B, shard=0, num_shards=1, choose_part=False):
        self.data_path_A = data_path_A
        self.data_path_B = data_path_B
        self.data_list_A = data_list_all_A[shard:][::num_shards]
        self.data_list_B = data_list_all_B[shard:][::num_shards]
        
        self.sz = 256
        self.has_label_B = has_label_B
        self.choose_part = choose_part
    
    def __len__(self,):
        return len(self.data_list_A) + len(self.data_list_B)
    
    def get_source_img(self, data_path, data_list, source):
        idx1 = np.random.randint(0, len(data_list) - 1)
        if not source and np.random.random() > 0.9:
            idx1 = np.random.randint(0, len(self.has_label_B) - 1)
            name_A = self.has_label_B[idx1]
        else:
            name_A = data_list[idx1]
        img_path = data_path + 'Original/' + name_A + '.png'
        lbl_path = data_path + 'Ground_truth/' + name_A + '_label.png'
        thick_label_path = data_path + 'Ground_truth/' + name_A + '_thick_label.png'
        
        arr = cv2.imread(img_path, 0)
        thick_lbl = cv2.imread(thick_label_path, 0)
        thick_lbl[thick_lbl == 255] = 1
        lbl = cv2.imread(lbl_path, 0)
        lbl[lbl == 255] = 1
        # if not source and name_A not in self.has_label_B:
        #     lbl[lbl == 1] = 0
        #     thick_lbl[thick_lbl == 1] = 0

        if arr.shape[0] > self.sz:
            if self.choose_part and source:
                left, right = lbl[:, :arr.shape[1] // 2].sum(), lbl[:, arr.shape[1] // 2:].sum()
                ch_x, ch_y = arr.shape[0] - self.sz - 1, arr.shape[1] - self.sz - 1
                xx_b = np.random.randint(0, ch_x)
                if left > right:
                    yy_b = np.random.randint(ch_y // 2, ch_y)
                else:
                    yy_b = np.random.randint(0, ch_y // 2)
            else:
                xx_b = np.random.randint(0, arr.shape[0] - self.sz - 1)
                yy_b = np.random.randint(0, arr.shape[1] - self.sz - 1)
            arr = arr[xx_b: xx_b + self.sz, yy_b: yy_b + self.sz]
            lbl = lbl[xx_b: xx_b + self.sz, yy_b: yy_b + self.sz]
            thick_lbl = thick_lbl[xx_b: xx_b + self.sz, yy_b: yy_b + self.sz]
        arr = arr.astype(np.float32) / 127.5 - 1
        return arr[np.newaxis, :, :], lbl[np.newaxis, :, :], thick_lbl[np.newaxis, :, :], name_A

    def __getitem__(self, idx):
        data_A, lbl_A, thick_A, name_A = self.get_source_img(self.data_path_A, self.data_list_A, source=True)
        data_B, lbl_B, thick_B, name_B = self.get_source_img(self.data_path_B, self.data_list_B, source=False)
        return {
            'data_A': data_A, 'data_B': data_B,
            'vessel_img_A': lbl_A, 'vessel_img_B': lbl_B,
            'thick_label_A': thick_A, 'thick_label_B': thick_B,
            'name_A': name_A, 'name_B': name_B,
        }

class RetainDatasetTarget2(Dataset):
    def __init__(self, data_path_A, data_path_B, data_list_all_A, data_list_all_B, has_label_B, shard=0, num_shards=1, choose_part=False, source_have_label=True):
        self.data_path_A = data_path_A
        self.data_path_B = data_path_B
        self.data_list_A = data_list_all_A[shard:][::num_shards]
        self.data_list_B = data_list_all_B[shard:][::num_shards]
        
        self.sz = 256
        self.has_label_B = has_label_B
        self.choose_part = choose_part
        self.source_have_label = source_have_label
    
    def __len__(self,):
        return len(self.data_list_A) + len(self.data_list_B)
    
    def get_source_img(self,):
        idx1 = np.random.randint(0, len(self.data_list_A) - 1)
        name_A = self.data_list_A[idx1]
        
        img_path = self.data_path_A + name_A + '_t0.png'
        lbl_path = self.data_path_A + name_A + '_label.png'
        thick_label_path = self.data_path_A + name_A + '_thick_label.png'
        arr = cv2.imread(img_path, 0)
        thick_lbl = cv2.imread(thick_label_path, 0)
        thick_lbl[thick_lbl == 255] = 1
        lbl = cv2.imread(lbl_path, 0)
        lbl[lbl == 255] = 1
        arr = arr.astype(np.float32) / 127.5 - 1
        
        if not self.source_have_label:
            lbl[lbl == 1] = 0
            thick_lbl[thick_lbl == 1] = 0
        
        return arr[np.newaxis, :, :], lbl[np.newaxis, :, :], thick_lbl[np.newaxis, :, :], name_A
    
    def get_target_img(self,):
        idx2 = np.random.randint(0, len(self.data_list_B) - 1)
        if np.random.random() > 0.9:
            idx2 = np.random.randint(0, len(self.has_label_B) - 1)
            name_A = self.has_label_B[idx2]
        else:
            name_A = self.data_list_B[idx2]
        img_path = self.data_path_B + 'Original/' + name_A + '.png'
        lbl_path = self.data_path_B + 'Ground_truth/' + name_A + '_label.png'
        thick_label_path = self.data_path_B + 'Ground_truth/' + name_A + '_thick_label.png'
        
        arr = cv2.imread(img_path, 0)
        thick_lbl = cv2.imread(thick_label_path, 0)
        thick_lbl[thick_lbl == 255] = 1
        lbl = cv2.imread(lbl_path, 0)
        lbl[lbl == 255] = 1
        if arr.shape[0] > self.sz:
            xx_b = np.random.randint(0, arr.shape[0] - self.sz - 1)
            yy_b = np.random.randint(0, arr.shape[1] - self.sz - 1)
            arr = arr[xx_b: xx_b + self.sz, yy_b: yy_b + self.sz]
            lbl = lbl[xx_b: xx_b + self.sz, yy_b: yy_b + self.sz]
            thick_lbl = thick_lbl[xx_b: xx_b + self.sz, yy_b: yy_b + self.sz]
        arr = arr.astype(np.float32) / 127.5 - 1
        return arr[np.newaxis, :, :], lbl[np.newaxis, :, :], thick_lbl[np.newaxis, :, :], name_A

    def __getitem__(self, idx):
        data_A, lbl_A, thick_A, name_A = self.get_source_img()
        data_B, lbl_B, thick_B, name_B = self.get_target_img()
        return {
            'data_A': data_A, 'data_B': data_B,
            'vessel_img_A': lbl_A, 'vessel_img_B': lbl_B,
            'thick_label_A': thick_A, 'thick_label_B': thick_B,
            'name_A': name_A, 'name_B': name_B,
        }

def load_data2(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    data_list_all = sorted(read_txt(f'{data_dir}train_ct.txt'))
    data_list_all_new = data_list_all * 4
    has_label = data_list_all[:8]
    dataset = TargetDataset(
        data_dir,
        data_list_all_new,
        has_label,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    
    print(dataset.__len__())
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
    while True:
        yield from loader

class TargetDataset(Dataset):

    def __init__(self, data_path, image_paths, has_label, shard=0, num_shards=1):
        self.data_path = data_path
        self.data_list = image_paths[shard:][::num_shards]
        self.has_label = has_label
    
    def __len__(self,):
        return len(self.data_list)
    
    def crop_patch(self, img, label, thick_label):
        # import pdb;pdb.set_trace()
        crop_size = (96, 128, 128)
        if np.random.random() > 0.5:
            xx_b = np.random.randint(0, img.shape[0] - crop_size[0] - 1)
            yy_b = np.random.randint(0, img.shape[1] - crop_size[1] - 1)
            zz_b = np.random.randint(0, img.shape[2] - crop_size[2] - 1)
        else:
            label_index = np.where(label == 1)
            rand_index = np.random.randint(0, label_index[0].shape[0])
            xx_c, yy_c, zz_c = label_index[0][rand_index], label_index[1][rand_index], label_index[2][rand_index]
            xx_b, yy_b, zz_b = get_bound(xx_c, crop_size[0], img.shape[0]), get_bound(yy_c, crop_size[1], img.shape[1]), get_bound(zz_c, crop_size[2], img.shape[2])
            
        img_patch = img[xx_b: xx_b + crop_size[0], yy_b: yy_b + crop_size[1], zz_b: zz_b + crop_size[2]]
        lbl_patch = label[xx_b: xx_b + crop_size[0], yy_b: yy_b + crop_size[1], zz_b: zz_b + crop_size[2]]
        thick_label_patch = thick_label[xx_b: xx_b + crop_size[0], yy_b: yy_b + crop_size[1], zz_b: zz_b + crop_size[2]]
        return img_patch, lbl_patch, thick_label_patch
    
    def __getitem__(self, idx):
        name = self.data_list[idx]
        label = np.load(self.data_path + 'size_ct_lbl/' + name + '_label.npy', mmap_mode='r')
        thick_label = np.load(self.data_path + 'vessel_center_ct/' + name + '_thick_label.npy', mmap_mode='r')
        data = np.load(self.data_path + 'size_ct_img/' + name + '_norm.npy', mmap_mode='r')
        # import pdb;pdb.set_trace()
        # data = (data - data.min()) / (data.max() - data.min()) * 2.0 - 1.0
        # print(data.max(), data.min())
        img_patch, lbl_patch, thick_label_patch = self.crop_patch(copy.deepcopy(data), copy.deepcopy(label), copy.deepcopy(thick_label))
        if name not in self.has_label:
            lbl_patch[lbl_patch == 1] = 0
            thick_label_patch[thick_label_patch == 1] = 0
        # print(img_patch.shape, lbl_patch.shape)
        return {'data': torch.from_numpy(img_patch)[None, ...], 'vessel_img': torch.from_numpy(lbl_patch)[None, ...],
                'thick_label': torch.from_numpy(thick_label_patch)[None, ...], 'name': name}

def nii2npy():
    root_path = '/data/guozhanqiang_07_temp/guozhanqiang/binary2gray/generation_test/generation_voxel/'
    name_list = sorted(os.listdir(root_path))
    # data_list_all = sorted(read_txt(f'/disk3/guozhanqiang/binary2gray/AllData/test.txt'))
    for name in name_list:
        if name[-7:] != '.nii.gz':
            continue
        # test_ = False
        # for need_name_one in data_list_all:
        #     if need_name_one in name:
        #         test_ = True
        #         break
        # if not test_:
        #     continue
        # out_name = (root_path + name).replace('.nii.gz', '_norm.npy')
        out_name = root_path + name.replace('.nii.gz', '.npy')
        print(name)
        img = sitk.GetArrayFromImage(sitk.ReadImage(root_path + name)).astype(np.uint8)
        # img = (img - img.min()) / (img.max() - img.min()) * 2.0 -1.0
        # img2 = np.load(out_name, mmap_mode='r')
        # print((img != img2).sum())
        np.save(out_name, img)

def vessel_map():
    root_path = '/data/guozhanqiang/binary2gray/fives/ROSE/'##OCTA-500
    img_path = root_path + 'Original/'
    lbl_path = root_path + 'Ground_truth/'
    name_list = sorted(os.listdir(img_path))

    for name in name_list:
        if '.png' not in name:
            continue
        # if name != 'TRV4P6CTAI.nii.gz':
        #     continue
        print(name)
        # out_thick_label = lbl_path + name.replace('.nii.gz', '_thick_label.nii.gz')
        # img_ = sitk.ReadImage(lbl_path + name.replace('.nii.gz', '_label.nii.gz'))
        # label_img = sitk.GetArrayFromImage(sitk.ReadImage(lbl_path + name.replace('.nii.gz', '_label.nii.gz')))
        out_thick_label = lbl_path + name.replace('.png', '_thick_label.png')
        label_img = cv2.imread(lbl_path + name.replace('.png', '_label.png'))
        label_img[label_img == 255] = 1

        distance_map = distance_transform_edt(1 - label_img)
        print((distance_map == 0).sum(), label_img.sum())
        thick_img = np.zeros(label_img.shape)
        thick_img[distance_map <= 2.0] = 1
        print((distance_map == 0).sum(), label_img.sum(), thick_img.sum())
        cv2.imwrite(out_thick_label, thick_img * 255)
        # thick_img_ = sitk.GetImageFromArray(thick_img)
        # thick_img_.CopyInformation(img_)
        # sitk.WriteImage(thick_img_, out_thick_label)

def vessel_map2():
    img_path = '/disk3/guozhanqiang/binary2gray/Improve-DDPM/openai-2025-01-15-16-49-11-714895/'
    name_list = sorted(os.listdir(img_path))

    for name in name_list:
        if '_label.png' not in name:
            continue
        print(name)
        # out_thick_label = lbl_path + name.replace('.nii.gz', '_thick_label.nii.gz')
        # img_ = sitk.ReadImage(lbl_path + name.replace('.nii.gz', '_label.nii.gz'))
        # label_img = sitk.GetArrayFromImage(sitk.ReadImage(lbl_path + name.replace('.nii.gz', '_label.nii.gz')))
        out_thick_label = img_path + name.replace('_label.png', '_thick_label.png')
        label_img = cv2.imread(img_path + name)
        label_img[label_img == 255] = 1

        distance_map = distance_transform_edt(1 - label_img)
        print((distance_map == 0).sum(), label_img.sum())
        thick_img = np.zeros(label_img.shape)
        thick_img[distance_map <= 2.0] = 1
        print((distance_map == 0).sum(), label_img.sum(), thick_img.sum())
        cv2.imwrite(out_thick_label, thick_img * 255)

def new_to():
    ori_path = '/disk3/guozhanqiang/binary2gray/Improve-DDPM/openai-2025-01-07-22-58-59-929243/'
    new_path = '/disk3/guozhanqiang/binary2gray/Improve-DDPM/usage_t0/'
    os.makedirs(new_path, exist_ok=True)
    name_list = sorted(os.listdir(ori_path))
    for name in name_list:
        print(name)
        if '.png' not in name:
            continue
        img = cv2.imread(ori_path + name, 0)
        cv2.imwrite(new_path + name[6:], img)

if __name__ == '__main__':
    # data_dir_A = "/disk3/guozhanqiang/binary2gray/fives/post_data/"
    # data_dir_B = "/disk3/guozhanqiang/binary2gray/fives/OCTA-500/"
    # data_list_all_A = sorted(read_txt(f'{data_dir_A}train.txt'))
    # data_list_all_B = sorted(read_txt(f'{data_dir_B}train.txt'))
    # has_label_B = data_list_all_B[:5]
    # dataset = RetainDatasetTarget(data_dir_A, data_dir_B, data_list_all_A, data_list_all_B, has_label_B, choose_part=True)
    # for ii in range(0, dataset.__len__()):
    #     if ii == 234:
    #         print(1234)
    #         import pdb;pdb.set_trace()
    #     data = dataset.__getitem__(ii)
    #     data_A, data_B = data['data_A'][0], data['data_B'][0]
    #     lbl_A, lbl_B = data['vessel_img_A'][0], data['vessel_img_B'][0]
    #     if ii == 234:
    #         if lbl_B.sum() > 0:
    #             cv2.imwrite(f'./temp/lbl_A_{ii}.png', lbl_A * 255.0)
    #             cv2.imwrite(f'./temp/lbl_B_{ii}.png', lbl_B * 255.0)
    #         print(lbl_A.sum(), lbl_B.sum())
    
    # vessel_map()
    nii2npy()
    # vessel_map2()
    # new_to()
    # data_dir = "/disk3/guozhanqiang/binary2gray/fives/post_data/"
    # data_list_all = sorted(read_txt(f'{data_dir}train.txt'))
    # dataset = RetainDataset(
    #     data_dir,
    #     data_list_all,
    #     choose_part=True
    # )

    # for ii in range(0, dataset.__len__()):
    #     data = dataset.__getitem__(ii)
    #     img, lbl = np.mean(data['data'], 0), data['vessel_img'][0]
    #     print(data['data'].shape, data['data'].min(), data['data'].max())
    #     print(data['vessel_img'].sum())
    #     if lbl.sum() != 0:
    #         print(img.mean(), (img * lbl).sum() / lbl.sum())
    #     cv2.imwrite(f'./temp/img_{ii}.png', (img + 1.0) / 2 * 255.0)
    #     cv2.imwrite(f'./temp/lbl_{ii}.png', lbl * 255.0)
