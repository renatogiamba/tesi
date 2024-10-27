import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from pathlib import Path
from torch.utils import data as data
import csv
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register(suffix='basicsr')
class FmowSentinelDataset(data.Dataset):
    """Modified dataset based on the dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(FmowSentinelDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        if 'crop_size' in opt:
            self.crop_size = opt['crop_size']
        else:
            self.crop_size = 512
        if 'image_type' not in opt:
            opt['image_type'] = 'jpg'

        # support multiple type of data: file path and meta data, remove support of lmdb
        self.paths = []
        if 'meta_info' in opt:
            with open(self.opt['meta_info']) as fin:
                    paths = [line.strip().split(' ')[0] for line in fin]
                    self.paths = [v for v in paths]
            if 'meta_num' in opt:
                self.paths = sorted(self.paths)[:opt['meta_num']]
        if 'gt_path' in opt:
            if isinstance(opt['gt_path'], str):
                self.paths.extend(sorted([str(x) for x in Path(opt['gt_path']).glob('*.'+opt['image_type'])]))
            else:
                self.paths.extend(sorted([str(x) for x in Path(opt['gt_path'][0]).glob('*.'+opt['image_type'])]))
                if len(opt['gt_path']) > 1:
                    for i in range(len(opt['gt_path'])-1):
                        self.paths.extend(sorted([str(x) for x in Path(opt['gt_path'][i+1]).glob('*.'+opt['image_type'])]))

        # limit number of pictures for test
        if 'num_pic' in opt:
            if 'val' or 'test' in opt:
                random.shuffle(self.paths)
                self.paths = self.paths[:opt['num_pic']]
            else:
                self.paths = self.paths[:opt['num_pic']]

        if 'mul_num' in opt:
            self.paths = self.paths * opt['mul_num']
            # print('>>>>>>>>>>>>>>>>>>>>>')
            # print(self.paths)

        self.lr_path = opt['lr_path']
        self.meta_info = get_all_csv_info_as_dict(opt['gt_path'])

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        # avoid errors caused by high latency in reading files
        retry = 10
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
                img_gt = imfrombytes(img_bytes, float32=True)
            except Exception as e:
                # logger = get_root_logger()
                # logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__()-1)
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
                print("retrying", e)
            else:
                break
            finally:
                retry -= 1

        lr_filename = os.path.basename(gt_path)
        lr_path = os.join(self.lr_path, lr_filename)
        img_lr = self.file_client.get(lr_path, 'lr')
        img_lr = imfrombytes(img_lr, float32=True)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        img_lr = img2tensor([img_lr], bgr2rgb=True, float32=True)[0]

        return_d = {'gt': img_gt,
                    'lr': img_lr,
                    'gt_path': gt_path}
        
        category, country, gsd, cloud_cover, year, month, day = self.get_metadata(index)
        text_prompt = self.create_sentence(category, country, year)
        return_d['gsd'] = gsd
        return_d['cloud_cover'] = cloud_cover
        return_d['year'] = year
        return_d['month'] = month
        return_d['day'] = day
        return_d['text_prompt'] = text_prompt
        return return_d


    def create_sentence(self, category, country, year):
        sentences = [
            "A high resolution fmow satellite image of a {} "+
           "in the state {} in the year {}.",
        ]
        sentence = random.choice(sentences)
        return sentence.format(category, country, year)

    def get_metadata(self, idx):
        path_name = self.paths[idx]
        category = self.meta_info[os.path.basename(path_name)[:-4]]['category']
        country = self.meta_info[os.path.basename(path_name)[:-4]]['country']
        gsd = self.meta_info[os.path.basename(path_name)[:-4]]['gsd']
        cloud_cover = self.meta_info[os.path.basename(path_name)[:-4]]['cloud_cover']
        year = self.meta_info[os.path.basename(path_name)[:-4]]['year']
        month = self.meta_info[os.path.basename(path_name)[:-4]]['month']
        day = self.meta_info[os.path.basename(path_name)[:-4]]['day']
        return category, country, gsd, cloud_cover, year, month, day

    def __len__(self):
        return len(self.paths)


def extract_info_from_csv(csv_file, mini_dict):
    with open(csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        # mini_dict = {}
        next(csv_reader)

        for row in csv_reader:
            mini_dict[row[0]] = {}
            # mini_dict = row[0]
            mini_dict[row[0]]['category'] = row[1]
            mini_dict[row[0]]['country'] = row[2]
            mini_dict[row[0]]['gsd'] = row[3]
            mini_dict[row[0]]['cloud_cover'] = row[4]
            mini_dict[row[0]]['year'] = row[5]
            mini_dict[row[0]]['month'] = row[6]
            mini_dict[row[0]]['day'] = row[7]
            line_count += 1

        # print(f'Processed {line_count} lines.')
        return mini_dict


#function to list all csv files in path and subpath
def list_all_csv_files(path):
    ls = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                ls.append(os.path.join(root, file))
    return ls

def get_all_csv_info_as_dict(path):
    csvs = list_all_csv_files(path)
    diz = {}
    for i in range(len(csvs)):
        diz = extract_info_from_csv(csvs[i], diz)
    return diz