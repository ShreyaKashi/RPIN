import phyre
import torch
import numpy as np
from glob import glob
import torch.nn.functional as F
# ---- NeuralPhys Helper Functions
from rpin.utils.config import _C as C
from rpin.utils.im import get_im_data
from rpin.utils.vis import plot_rollouts, WorldToImg_coord
from rpin.utils.misc import tprint, pprint
from rpin.utils.bbox import xyxy_to_rois, xywh2xyxy
from rpin.util import to_device
import pickle
import cv2
import copy

MCS_IMG_HEIGHT = 600
MCS_IMG_WIDTH = 400

class PredEvaluator(object):
    def __init__(self, device, val_loader, model, num_gpus, num_plot_image, output_dir):
        # misc
        self.device = device
        self.output_dir = output_dir
        self.num_gpus = num_gpus
        self.plot_image = num_plot_image
        # data loader
        self.val_loader = val_loader
        # nn
        self.model = model
        # input setting
        self.ptrain_size, self.ptest_size = C.RPIN.PRED_SIZE_TRAIN, C.RPIN.PRED_SIZE_TEST
        self.input_height, self.input_width = C.RPIN.INPUT_HEIGHT, C.RPIN.INPUT_WIDTH
        # loss settings
        self._setup_loss()
        self.high_resolution_plot = True
        self.vae_num_samples = 100

    def test(self):
        self.model.eval()

        plot_help=0
        for batch_idx, (features, pointclouds, edges_self, edges_forward, data_pc_oind, data_pc_oind_help, data_pc_find, data_pc_bind, gt_center3d_world, valid, g_idx, labels) in enumerate(self.val_loader):
            with torch.no_grad():

                features = to_device(features)
                pointclouds = to_device(pointclouds)
                edges_self = to_device(edges_self)
                edges_forward = to_device(edges_forward)
                data_pc_oind = to_device(data_pc_oind)
                data_pc_oind_help = to_device(data_pc_oind_help)
                data_pc_find = to_device(data_pc_find)
                data_pc_bind = to_device(data_pc_bind)

                labels = {
                'gt_center3d_world': to_device(gt_center3d_world) if self.device == torch.device('cuda') else gt_center3d_world,
                'valid': to_device(valid) if self.device == torch.device('cuda') else gt_center3d_world,
                }

                outputs = self.model(features, pointclouds, edges_self, edges_forward, data_pc_oind, data_pc_oind_help, data_pc_find, data_pc_bind, num_rollouts=self.ptest_size, g_idx=g_idx, phase='test')
                self.loss(outputs, labels, 'test')

                tprint(f'eval: {batch_idx}/{len(self.val_loader)}:' + ' ' * 20)

            if self.plot_image > 0:
                
                cam_extrinsic_mat=pickle.load(open('./mcs_preprocess/cam_params/cam_extrinsic_mat.pkl', "rb"))
                cam_intrinsic_mat=pickle.load(open('./mcs_preprocess/cam_params/cam_intrinsic_mat.pkl', "rb"))

                outputs = {
                    'center3d_world': outputs['center3d_world'].cpu().numpy(),
                    # 'masks': outputs['masks'].cpu().numpy() if C.RPIN.MASK_LOSS_WEIGHT else None,
                    # 'center3d_2d_offset': outputs['center3d_2d_offset'].cpu().numpy() if C.RPIN.CENTER3D_2D_OFFSET_LOSS_WEIGHT else None,
                    # 'center3d_2d_depth': outputs['center3d_2d_depth'].cpu().numpy() if C.RPIN.CENTER3D_2D_DEPTH_LOSS_WEIGHT else None,
                }
                # outputs['boxes'][..., 0::2] *= self.input_width
                # outputs['boxes'][..., 1::2] *= self.input_height

                # output_box_2dcenter=outputs['boxes'][..., 0:2].copy()

                # outputs['boxes'] = xywh2xyxy(
                #     outputs['boxes'].reshape(-1, 4)
                # ).reshape((data.shape[0], -1, C.RPIN.MAX_NUM_OBJS, 4))

                # outputs['center3d_2d_offset'][..., 0] *= self.input_width
                # outputs['center3d_2d_offset'][..., 1] *= self.input_height
                # if C.RPIN.CENTER3D_2D_INVERSE_DEPTH == True:
                #     outputs['center3d_2d_offset'][..., 2] = 1/outputs['center3d_2d_offset'][..., 2]

                labels = {
                    'gt_center3d_world' : labels['gt_center3d_world'].cpu().numpy(),
                    # 'boxes': labels['boxes'].cpu().numpy(),
                    # 'masks': labels['masks'].cpu().numpy(),
                    # 'center3d_2d': labels['center3d_2d'].cpu().numpy(),
                    # 'center3d_2d_offset': labels['center3d_2d_offset'].cpu().numpy(),
                }
                # labels['boxes'][..., 0::2] *= self.input_width
                # labels['boxes'][..., 1::2] *= self.input_height
                # labels['boxes'] = xywh2xyxy(
                #     labels['boxes'].reshape(-1, 4)
                # ).reshape((data.shape[0], -1, C.RPIN.MAX_NUM_OBJS, 4))

                # labels['center3d_2d'][..., 0] *= self.input_width
                # labels['center3d_2d'][..., 1] *= self.input_height
                # if C.RPIN.CENTER3D_2D_INVERSE_DEPTH == True:
                #     labels['center3d_2d'][..., 2] = 1/labels['center3d_2d'][..., 2]

                for i in range(valid.shape[0]):
                    batch_size = C.SOLVER.BATCH_SIZE 
                    plot_image_idx = batch_size * batch_idx + i
                    # if plot_image_idx < self.plot_image:
                    if plot_help < self.plot_image:
                        tprint(f'plotting: {plot_image_idx}' + ' ' * 20)
                        video_idx, img_idx = self.val_loader.dataset.video_pc_info[plot_image_idx]
                        if img_idx !=0:
                            continue

                        video_name = self.val_loader.dataset.video_pc_list[video_idx].replace('_pc', '')
                        # print('video_name',video_name)
                        # print('img_idx',img_idx)

                        v = valid[i].numpy().astype(bool)
                        pred_center3d_world_i = outputs['center3d_world'][i][:, v]
                        gt_center3d_world_i = labels['gt_center3d_world'][i][:, v]

                        pred_center3d_world_img_i,pred_center3d_world_dep_i=WorldToImg_coord(pred_center3d_world_i, cam_extrinsic_mat, cam_intrinsic_mat)
                        gt_center3d_world_img_i,gt_center3d_world_dep_i=WorldToImg_coord(gt_center3d_world_i, cam_extrinsic_mat, cam_intrinsic_mat)

                        # pred_boxes_i = outputs['boxes'][i][:, v]
                        # gt_boxes_i = labels['boxes'][i][:, v]

                        # if 'PHYRE' in C.DATA_ROOT:
                        #     im_data = phyre.observations_to_float_rgb(np.load(video_name).astype(np.uint8))[..., ::-1]
                        #     a, b, c = video_name.split('/')[-3:]
                        #     output_name = f'{a}_{b}_{c.replace(".npy", "")}'

                        #     bg_image = np.load(video_name).astype(np.uint8)
                        #     for fg_id in [1, 2, 3, 5]:
                        #         bg_image[bg_image == fg_id] = 0
                        #     bg_image = phyre.observations_to_float_rgb(bg_image)
                        # else:
                        bg_image = None
                        image_list = sorted(glob(f'{video_name}/*.png'))
                        # print('image_list',image_list)
                        resized_im_list = []
                        for pre_size in range(self.ptest_size):
                            im_name = image_list[img_idx+C.RPIN.INPUT_SIZE+pre_size]
                            # deal with image data here
                            # gt_boxes_i = labels['boxes'][i][:, v]
                            im_data = get_im_data(im_name, None, C.DATA_ROOT, self.high_resolution_plot)

                            resized_im_data = cv2.resize(im_data, (MCS_IMG_HEIGHT, MCS_IMG_WIDTH))
                            resized_im_list.append(resized_im_data)
                        output_name = '_'.join(image_list[img_idx].split('.')[0].split('/')[-2:])
                        # print('resized_im_data.size',resized_im_data.shape)
                        # gt_center3d_2d_center_depth_i = labels['center3d_2d'][i][:, v]
                        # # pred_center3d_2d_offset_i = None
                        # pred_center3d_2d_center_i = None
                        # if C.RPIN.CENTER3D_2D_OFFSET_LOSS_WEIGHT:
                        #     # pred_center3d_2d_offset_i = outputs['center3d_2d_offset'][i][:, v]
                        #     pred_center3d_2d_center_i = outputs['center3d_2d_offset'][i][:, v] + output_box_2dcenter[i][:, v]

                        # pred_center3d_2d_depth_i = None
                        # if C.RPIN.CENTER3D_2D_DEPTH_LOSS_WEIGHT:
                        #     pred_center3d_2d_depth_i = outputs['center3d_2d_depth'][i][:, v]

                        # if self.high_resolution_plot:
                        #     scale_w = im_data.shape[1] / self.input_width
                        #     scale_h = im_data.shape[0] / self.input_height
                        #     pred_boxes_i[..., [0, 2]] *= scale_w
                        #     pred_boxes_i[..., [1, 3]] *= scale_h
                        #     gt_boxes_i[..., [0, 2]] *= scale_w
                        #     gt_boxes_i[..., [1, 3]] *= scale_h
                        #     if C.RPIN.CENTER3D_2D_OFFSET_LOSS_WEIGHT:
                        #         pred_center3d_2d_center_i[..., [0]] *= scale_w
                        #         pred_center3d_2d_center_i[..., [1]] *= scale_h
                        #         gt_center3d_2d_center_depth_i[..., [0]] *= scale_w
                        #         gt_center3d_2d_center_depth_i[..., [1]] *= scale_h
                        # assert 1 == 2

                        # pred_masks_i = None
                        # if C.RPIN.MASK_LOSS_WEIGHT:
                        #     pred_masks_i = outputs['masks'][i][:, v]

                        plot_rollouts(resized_im_list, pred_center3d_world_img_i,gt_center3d_world_img_i,
                                      pred_center3d_world_dep=pred_center3d_world_dep_i, gt_center3d_world_dep=gt_center3d_world_dep_i, 
                                      output_dir=self.output_dir, output_name=output_name, bg_image=bg_image)
                        plot_help+=1
                # assert 1 == 2
        print('\r', end='')
        print_msg = ""
        mean_loss = np.mean(np.array(self.center3d_world_step_losses[:self.ptest_size]) / self.loss_cnt) * 1e3
        print_msg += f"{mean_loss:.3f} | "
        print_msg += f" | ".join(["{:.3f}".format(self.losses[name] * 1e3 / self.loss_cnt) for name in self.loss_name])
        if C.RPIN.SEQ_CLS_LOSS_WEIGHT:
            print_msg += f" | {self.fg_correct / (self.fg_num + 1e-9):.3f} | {self.bg_correct / (self.bg_num + 1e-9):.3f}"
        pprint(print_msg)

    def loss(self, outputs, labels, phase):
        self.loss_cnt += labels['gt_center3d_world'].shape[0]
        pred_size = eval(f'self.p{phase}_size')
        # calculate bbox loss
        # of shape (batch, time, #obj, 4)
        loss = (outputs['center3d_world'] - labels['gt_center3d_world']) ** 2
        # take weighted sum over axis 2 (objs dim) since some index are not valid
        valid = labels['valid'][:, None, :, None]
        loss = loss * valid
        loss = loss.sum(2) / valid.sum(2)
        loss *= self.position_loss_weight

        for i in range(pred_size):
            self.center3d_world_step_losses[i] += loss[:, i, :].sum().item()

        self.losses['p_1'] = float(np.mean(self.center3d_world_step_losses[:self.ptrain_size]))
        self.losses['p_2'] = float(np.mean(self.center3d_world_step_losses[self.ptrain_size:])) \
            if self.ptrain_size < self.ptest_size else 0
        
        # if C.RPIN.MASK_LOSS_WEIGHT > 0:
        #     # of shape (batch, time, #obj, m_sz, m_sz)
        #     mask_loss_ = F.binary_cross_entropy(outputs['masks'], labels['masks'], reduction='none')
        #     mask_loss = mask_loss_.mean((3, 4))
        #     valid = labels['valid'][:, None, :]
        #     mask_loss = mask_loss * valid
        #     mask_loss = mask_loss.sum(2) / valid.sum(2)
        #     for i in range(pred_size):
        #         self.masks_step_losses[i] += mask_loss[:, i].sum().item()

        #     m1_loss = self.masks_step_losses[:self.ptrain_size]
        #     m2_loss = self.masks_step_losses[self.ptrain_size:] if self.ptrain_size < self.ptest_size else 0
        #     self.losses['m_1'] = np.mean(m1_loss)
        #     self.losses['m_2'] = np.mean(m2_loss)

        # if C.RPIN.SEQ_CLS_LOSS_WEIGHT > 0:
        #     seq_loss = F.binary_cross_entropy(outputs['score'], labels['seq_l'], reduction='none')
        #     self.losses['seq'] += seq_loss.sum().item()
        #     # calculate accuracy
        #     s = (outputs['score'] >= 0.5).eq(labels['seq_l'])
        #     fg_correct = s[labels['seq_l'] == 1].sum().item()
        #     bg_correct = s[labels['seq_l'] == 0].sum().item()
        #     fg_num = (labels['seq_l'] == 1).sum().item()
        #     bg_num = (labels['seq_l'] == 0).sum().item()
        #     self.fg_correct += fg_correct
        #     self.bg_correct += bg_correct
        #     self.fg_num += fg_num
        #     self.bg_num += bg_num

        # if C.RPIN.CENTER3D_2D_OFFSET_LOSS_WEIGHT > 0:
        #     center3d_2d_offset_loss=(outputs['center3d_2d_offset'] - labels['center3d_2d_offset'][...,:2]) ** 2
        #     valid = labels['valid'][:, None, :, None]
        #     center3d_2d_offset_loss = center3d_2d_offset_loss * valid
        #     center3d_2d_offset_loss = center3d_2d_offset_loss.sum(2) / valid.sum(2)
        #     for i in range(pred_size):
        #         self.center3d_2d_o_step_losses[i] += center3d_2d_offset_loss[:, i, :].sum().item()

        #     self.losses['3d2d_o1'] = float(np.mean(self.center3d_2d_o_step_losses[:self.ptrain_size]))
        #     self.losses['3d2d_o2'] = float(np.mean(self.center3d_2d_o_step_losses[self.ptrain_size:])) \
        #         if self.ptrain_size < self.ptest_size else 0

        # if C.RPIN.CENTER3D_2D_DEPTH_LOSS_WEIGHT > 0:
        #     center3d_2d_depth_loss=(outputs['center3d_2d_depth'] - labels['center3d_2d_offset'][...,2:]) ** 2
        #     valid = labels['valid'][:, None, :, None]
        #     center3d_2d_depth_loss = center3d_2d_depth_loss * valid
        #     center3d_2d_depth_loss = center3d_2d_depth_loss.sum(2) / valid.sum(2)
        #     for i in range(pred_size):
        #         self.center3d_2d_d_step_losses[i] += center3d_2d_depth_loss[:, i, :].sum().item()

        #     self.losses['3d2d_d1'] = float(np.mean(self.center3d_2d_d_step_losses[:self.ptrain_size]))
        #     self.losses['3d2d_d2'] = float(np.mean(self.center3d_2d_d_step_losses[self.ptrain_size:])) \
        #         if self.ptrain_size < self.ptest_size else 0
            
        # if C.RPIN.CENTER3D_2D_INVERSE_DEPTH == True:
        #     center3d_2d_true_depth_loss=(1/outputs['center3d_2d_depth'] - 1/labels['center3d_2d_offset'][...,2:]) ** 2
        #     valid = labels['valid'][:, None, :, None]
        #     center3d_2d_true_depth_loss = center3d_2d_true_depth_loss * valid
        #     center3d_2d_true_depth_loss = center3d_2d_true_depth_loss.sum(2) / valid.sum(2)

        #     for i in range(pred_size):
        #         self.center3d_2d_true_d_step_losses[i] += center3d_2d_true_depth_loss[:, i, :].sum().item()

        #     self.losses['true_d1'] = float(np.mean(self.center3d_2d_true_d_step_losses[:self.ptrain_size]))
        #     self.losses['true_d2'] = float(np.mean(self.center3d_2d_true_d_step_losses[self.ptrain_size:])) \
        #         if self.ptrain_size < self.ptest_size else 0

        return

    def _setup_loss(self):
        self.loss_name = []
        self.position_loss_weight = C.RPIN.POSITION_LOSS_WEIGHT
        self.loss_name += ['p_1', 'p_2']
        # if C.RPIN.MASK_LOSS_WEIGHT:
        #     self.loss_name += ['m_1', 'm_2']
        # if C.RPIN.CENTER3D_2D_OFFSET_LOSS_WEIGHT:
        #     self.center3d_2d_offset_loss_weight = C.RPIN.CENTER3D_2D_OFFSET_LOSS_WEIGHT
        #     self.loss_name += ['3d2d_o1', '3d2d_o2']
        # if C.RPIN.CENTER3D_2D_DEPTH_LOSS_WEIGHT:
        #     self.center3d_2d_depth_loss_weight = C.RPIN.CENTER3D_2D_DEPTH_LOSS_WEIGHT
        #     self.loss_name += ['3d2d_d1', '3d2d_d2']
        # if C.RPIN.CENTER3D_2D_INVERSE_DEPTH == True: 
        #     self.loss_name += ['true_d1', 'true_d2']
        # if C.RPIN.SEQ_CLS_LOSS_WEIGHT:
        #     self.loss_name += ['seq']
        self._init_loss()

    def _init_loss(self):
        self.losses = dict.fromkeys(self.loss_name, 0.0)
        self.center3d_world_step_losses = [0.0 for _ in range(self.ptest_size)]
        # self.box_p_step_losses = [0.0 for _ in range(self.ptest_size)]
        # self.box_s_step_losses = [0.0 for _ in range(self.ptest_size)]
        # self.masks_step_losses = [0.0 for _ in range(self.ptest_size)]
        # self.center3d_2d_o_step_losses = [0.0 for _ in range(self.ptest_size)]
        # self.center3d_2d_d_step_losses = [0.0 for _ in range(self.ptest_size)]
        # self.center3d_2d_true_d_step_losses = [0.0 for _ in range(self.ptest_size)]
        # # an statistics of each validation
        # self.fg_correct, self.bg_correct, self.fg_num, self.bg_num = 0, 0, 0, 0
        self.loss_cnt = 0
