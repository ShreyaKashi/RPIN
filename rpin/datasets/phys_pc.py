import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from copy import deepcopy

from rpin.utils.config import _C as C
from rpin.utils.bbox import xyxy2xywh
from rpin.datasets.pc_common import subsample_and_knn
import yaml

plot = False
debug = False


class Phys_pc(Dataset):
    def __init__(self, data_root, split, image_ext='.jpg'):
        self.data_root = data_root
        self.split = split
        self.image_ext = image_ext
        # 1. define property of input and rollout parameters
        self.input_size = C.RPIN.INPUT_SIZE  # number of input images
        self.pred_size = eval(f'C.RPIN.PRED_SIZE_{"TRAIN" if split == "train" else "TEST"}')
        self.seq_size = self.input_size + self.pred_size
        # 2. define model configs
        self.input_height, self.input_width = C.RPIN.INPUT_HEIGHT, C.RPIN.INPUT_WIDTH
        self.depth_normalize = C.RPIN.DEPTH_NORMALIZE
        self.video_pc_list, self.anno_list = None, None
        self.video_pc_info = None
        self.subsample_and_knn_cfg = yaml.safe_load(open(C.RPIN.PCF_ARGS, 'r'))
        self.use_grid_level = 1

    def __len__(self):
        return self.video_pc_info.shape[0]

    def __getitem__(self, idx):
        vid_idx, img_idx = self.video_pc_info[idx, 0], self.video_pc_info[idx, 1]
        video_pc_name, anno_name = self.video_pc_list[vid_idx], self.anno_list[vid_idx]
        data_pc_rgbd, data_pc_ind, data_pc_find = self._parse_image(video_pc_name, vid_idx, img_idx)
        
        print('data_pc_rgbd',data_pc_rgbd.shape[0])
        print("data_pc_find",data_pc_find)
        print("data_pc_ind",data_pc_ind)

        center3d_real = self._parse_label(anno_name, vid_idx, img_idx)

        all_data = {}

        # image flip augmentation
        # if random.random() > 0.5 and self.split == 'train' and C.RPIN.HORIZONTAL_FLIP:
        #     boxes[..., [0, 2]] = self.input_width - boxes[..., [2, 0]]
        #     data = np.ascontiguousarray(data[..., ::-1])
        #     gt_masks = np.ascontiguousarray(gt_masks[..., ::-1])
        #     center3d_real[..., [0]] = self.input_width - center3d_real[..., [0]]

        # if random.random() > 0.5 and self.split == 'train' and C.RPIN.VERTICAL_FLIP:
        #     boxes[..., [1, 3]] = self.input_height - boxes[..., [3, 1]]
        #     data = np.ascontiguousarray(data[..., ::-1, :])
        #     gt_masks = np.ascontiguousarray(gt_masks[..., ::-1])
        #     center3d_real[..., [1]] = self.input_height - center3d_real[..., [1]]
        # data_pc_d = data_pc_rgbd[:,:3].copy()

        coord = data_pc_rgbd[:, :3]
        point_color = data_pc_rgbd[:, 3:]
        norm = None

        z_min = coord[:, 2].min()
        coord[:, 2] -= z_min

        coord_min = coord.min(0)
        coord -= coord_min

        num_objs = data_pc_ind.shape[1] - 1
        g_idx = []
        for i in range(C.RPIN.MAX_NUM_OBJS):
            for j in range(C.RPIN.MAX_NUM_OBJS):
                if j == i:
                    continue
                g_idx.append([i, j, (i < num_objs) * (j < num_objs)])
        g_idx = np.array(g_idx)

        point_list_all, nei_forward_list_all, nei_propagate_list_all, nei_self_list_all, norm_list_all = [] ,[] ,[] ,[] ,[]
        for i in range(data_pc_find.shape[0]-1):
            # print("i",i)
            # print('coord[data_pc_find[i]:data_pc_find[i+1],:]',coord[data_pc_find[i]:data_pc_find[i+1],:].shape)
            point_list, nei_forward_list, nei_propagate_list, nei_self_list, norm_list = \
                subsample_and_knn(coord[data_pc_find[i]:data_pc_find[i+1],:], norm, grid_size=self.subsample_and_knn_cfg["grid_size"], K_self=self.subsample_and_knn_cfg["K_self"],
                              K_forward=self.subsample_and_knn_cfg["K_forward"], K_propagate=self.subsample_and_knn_cfg["K_propagate"])
            # print('point_list',len(point_list))
            # print('nei_forward_list',len(nei_forward_list))
            # print('nei_self_list',len(nei_self_list))
            # point_list_all.append(point_list)
            # nei_forward_list_all.append(nei_forward_list)
            # # nei_propagate_list_all.append(nei_propagate_list)
            # nei_self_list_all.append(nei_self_list)
            # norm_list_all.append(norm_list)
            point_list_all += point_list
            nei_forward_list_all +=  nei_forward_list
            nei_self_list_all += nei_self_list

        # gt 3dcenter real
        gt_center3d_real=center3d_real[self.input_size:].copy()
        gt_center3d_real = gt_center3d_real.reshape(self.pred_size, -1, 3)

        labels = torch.zeros(1)  # a fake variable used to make interface consistent
        gt_center3d_real = torch.from_numpy(gt_center3d_real.astype(np.float32))

        all_data['label_list'] = labels
        all_data['feature_list'] = [point_color]
        all_data['point_list'] = point_list_all
        all_data['nei_forward_list'] = nei_forward_list_all
        # all_data['nei_propagate_list'] = nei_propagate_list_all
        all_data['nei_self_list'] = nei_self_list_all
        # all_data['surface_normal_list'] = norm_list

        all_data['data_pc_ind'] = data_pc_ind
        all_data['data_pc_find'] = data_pc_find
        all_data['gt_center3d_real'] = gt_center3d_real
        all_data['g_idx'] = g_idx

        # print('data_pc_ind',data_pc_ind.shape)
        # print('data_pc_find',data_pc_find.shape)

        # assert 1==2

        return all_data

    def _parse_image(self, video_pc_name, vid_idx, img_idx):
        raise NotImplementedError

    def _parse_label(self, anno_name, vid_idx, img_idx):
        raise NotImplementedError
