import cv2
import torch
import pickle
import numpy as np
from glob import glob

from rpin.datasets.phys_pc import Phys_pc
from rpin.utils.misc import tprint
from rpin.utils.config import _C as C


class SS_PC(Phys_pc):
    def __init__(self, data_root, split, image_ext='.jpg'):
        super().__init__(data_root, split, image_ext)

        # self.video_pc_list = sorted(glob(f'{self.data_root}/{self.split}/*/'))
        # if C.RPIN.DELET_UNFOCUSED:
        #     self.video_pc_list = sorted([folder for folder in glob(f'{self.data_root}/{self.split}/*/') if "occulder" in folder and "depth" not in folder])
        #     self.anno_list = [v[:-1].replace("_occulder","") + '_boxes.pkl' for v in self.video_pc_list]
        # else:
        #     self.video_pc_list = sorted([folder for folder in glob(f'{self.data_root}/{self.split}/*/') if "occulder" not in folder and "depth" not in folder])
        #     self.anno_list = [v[:-1] + '_boxes.pkl' for v in self.video_pc_list]
        # print(self.data_root)
        self.video_pc_list = sorted([folder for folder in glob(f'{self.data_root}/{self.split}/*/') if "_pc" in folder and "_pc_ind" not in folder])
        # self.anno_list = [v[:-1] + '_3dcenter_real.pkl' for v in self.video_pc_list]
        # print(self.video_pc_list)
        self.anno_list = [v[:-1] + '_find.pkl' for v in self.video_pc_list]

        self.video_pc_info = np.zeros((0, 2), dtype=np.int32)
        for idx, video_pc_name in enumerate(self.video_pc_list):
            tprint(f'loading progress: {idx}/{len(self.video_pc_list)}')
            num_im = len(glob(f'{video_pc_name}/*{image_ext}'))
            # In ShapeStack, we only use the sequence starting from the first frame
            # num_sw = min(1, num_im - self.seq_size + 1)
            num_sw = num_im - self.seq_size + 1
            if num_sw <= 0:
                continue
            video_pc_info_t = np.zeros((num_sw, 2), dtype=np.int32)
            video_pc_info_t[:, 0] = idx  # video index
            video_pc_info_t[:, 1] = np.arange(num_sw)  # sliding window index
            self.video_pc_info = np.vstack((self.video_pc_info, video_pc_info_t))

    def _parse_image(self, video_pc_name, vid_idx, img_idx):
        image_pc_list = sorted(glob(f'{video_pc_name}/*{self.image_ext}'))

        image_pc_ind_folder_name = video_pc_name[:-1] + "_ind"
        image_pc_ind_list = sorted(glob(f'{image_pc_ind_folder_name}/*{self.image_ext}'))

        image_pc_list = image_pc_list[img_idx:img_idx + self.input_size]
        image_pc_ind_list = image_pc_ind_list[img_idx:img_idx + self.input_size]

        data_pc_rgbd = torch.cat([
            pickle.load(open(image_pc_name, "rb")) for image_pc_name in image_pc_list
        ], 0).numpy().astype(np.float32)

        data_pc_ind = torch.cat([
            pickle.load(open(image_pc_name, "rb")) for image_pc_name in image_pc_ind_list
        ], 0).numpy()

        data_pc_find = pickle.load(open(video_pc_name[:-1]+"_find.pkl", "rb"))[img_idx:img_idx + self.input_size + 1]
        data_pc_find = (data_pc_find - torch.min(data_pc_find) ).numpy().T
        # data_pc_ind = np.expand_dims(data_pc_ind, axis=1)

        # for c in range(3):
        #     data_pc_rgbd[:, c] -= C.INPUT.IMAGE_MEAN[c]
        #     data_pc_rgbd[:, c] /= C.INPUT.IMAGE_STD[c]
        
        # for c in range(1):
        #     data_pc_ind[:, c] -= np.mean(C.INPUT.IMAGE_MEAN)
        #     data_pc_ind[:, c] /= np.mean(C.INPUT.IMAGE_STD)

        # combined_data = np.concatenate([data_pc_rgbd, data_pc_ind], 1)

        # image_pc_list = [sorted(glob(f'{video_pc_name}/*{self.image_ext}'))[img_idx + self.seq_size - 1]]
        # data_t = np.array([
        #     cv2.imread(image_name) for image_name in image_pc_list
        # ], dtype=float).transpose((0, 3, 1, 2))
        # for c in range(3):
        #     data_t[:, c] -= C.INPUT.IMAGE_MEAN[c]
        #     data_t[:, c] /= C.INPUT.IMAGE_STD[c]

        return data_pc_rgbd, data_pc_ind, data_pc_find

    def _parse_label(self, anno_name, vid_idx, img_idx):
        anno_name = anno_name.replace('pc_find.', '3dcenter_real.')
        with open(anno_name, 'rb') as f:
            center3d_real = pickle.load(f)[img_idx:img_idx + self.seq_size, :, 1:]

        # with open(anno_name, 'rb') as f:
        #     boxes = pickle.load(f)[img_idx:img_idx + self.seq_size, :, 1:]
        # gt_masks = np.zeros((self.pred_size, boxes.shape[1], C.RPIN.MASK_SIZE, C.RPIN.MASK_SIZE))
        # if C.RPIN.MASK_LOSS_WEIGHT > 0:
        #     anno_name = anno_name.replace('boxes.', 'masks.')
        #     with open(anno_name, 'rb') as f:
        #         gt_masks = pickle.load(f)
        #     gt_masks = gt_masks[img_idx:img_idx + self.seq_size].astype(np.float32)
        #     gt_masks = gt_masks[self.input_size:]

        # if C.RPIN.CENTER3D_2D_OFFSET_LOSS_WEIGHT > 0 or C.RPIN.CENTER3D_2D_DEPTH_LOSS_WEIGHT > 0:
        #     anno_name = anno_name.replace('masks.', '3dcenter_2d.')
        #     with open(anno_name, 'rb') as f:
        #         center3d_2d = pickle.load(f)[img_idx:img_idx + self.seq_size, :, 1:]

        return center3d_real
