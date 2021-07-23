r"""Unreal custom dataset"""

import os
import json

from PIL import Image
import pandas as pd
import numpy as np
import torch

from .dataset import CorrespondenceDataset

class UnrealDataset(CorrespondenceDataset):
    r"""Inherits CorrespondenceDataset"""
    def __init__(self, benchmark, datapath, thres, device, split):
        super(UnrealDataset, self).__init__(benchmark, datapath, thres, device, split)

        # train_data : columns - [src_name, trg_name, annotation_name]
        self.train_data = pd.read_csv(self.spt_path)
        self.src_imnames = np.array(self.train_data.iloc[:, 0])
        self.trg_imnames = np.array(self.train_data.iloc[:, 1])
        self.ann_names = np.array(self.train_data.iloc[:, 2])
        self.cls = os.listdir(self.img_path)  # class
        self.cls.sort()

        self.ann_names = list(map(lambda x: self.ann_path+'/'+x, self.ann_names))
        anntn_files = list(map(lambda x: json.load(open(x)), self.ann_names))
        self.src_kps = list(map(lambda x: torch.tensor(x['src_kps']).t().float(), anntn_files))
        self.trg_kps = list(map(lambda x: torch.tensor(x['trg_kps']).t().float(), anntn_files))
        self.cls_ids = list(map(lambda x: self.cls.index(x['category']), anntn_files))

    def __getitem__(self, idx):
        batch = super(UnrealDataset, self).__getitem__(idx)

        batch['pckthres'] = self.get_pckthres(batch, batch['trg_imsize'])

        batch['src_kpidx'] = self.match_idx(batch['src_kps'], batch['n_pts'])
        batch['trg_kpidx'] = self.match_idx(batch['trg_kps'], batch['n_pts'])

        return batch

    def get_image(self, img_names, idx):
        path = os.path.join(self.img_path, img_names[idx])

        return Image.open(path).convert('RGB')
