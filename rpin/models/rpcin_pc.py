# Written by Haozhi Qi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.roi_align import RoIAlign
import yaml
from easydict import EasyDict as edict

from rpin.utils.config import _C as C
from rpin.models.layers.CIN import InterNet
from rpin.models.backbones.build import build_backbone, build_pc_backbone
from rpin.models.backbones.pcn import get_default_configs

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # define private variables
        self.time_step = C.RPIN.INPUT_SIZE
        # self.ve_feat_dim = C.RPIN.VE_FEAT_DIM  # visual encoder feature dimension
        self.in_feat_dim = C.RPIN.IN_FEAT_DIM  # interaction net feature dimension
        # print('self.in_feat_dim',self.in_feat_dim)
        self.num_objs = C.RPIN.MAX_NUM_OBJS
        self.mask_size = C.RPIN.MASK_SIZE
        self.picked_state_list = [0, 3, 6, 9]

        # build image encoder
        # self.backbone = build_backbone(C.RPIN.BACKBONE, self.ve_feat_dim, C.INPUT.IMAGE_CHANNEL)
        # build point cloud encoder
        pc_backbon_cfg = edict(yaml.safe_load(open(C.RPIN.PCF_ARGS, 'r')))
        # print(pc_backbon_cfg)
        pc_backbon_cfg = get_default_configs(pc_backbon_cfg,num_level=pc_backbon_cfg['num_level'])
        # pc_backbon_cfg.pretrain_path = args.pretrain_path
        # pc_backbon_cfg.config = args.config
        # pc_backbon_cfg.split = args.split
        self.backbone = build_pc_backbone(pc_backbon_cfg)

        # extract object feature -> convert to object state
        pool_size = C.RPIN.ROI_POOL_SIZE
        self.roi_align = RoIAlign(
            (pool_size, pool_size),
            spatial_scale=C.RPIN.ROI_POOL_SPATIAL_SCALE,
            sampling_ratio=C.RPIN.ROI_POOL_SAMPLE_R,
        )

        # roi2state = [nn.Conv2d(self.ve_feat_dim, self.in_feat_dim, kernel_size=3, padding=1),
        #              nn.ReLU(inplace=True)]

        # for _ in range(C.RPIN.N_EXTRA_ROI_F):
        #     roi2state.append(nn.Conv2d(self.ve_feat_dim, self.in_feat_dim,
        #                                kernel_size=C.RPIN.EXTRA_F_KERNEL, stride=1, padding=C.RPIN.EXTRA_F_PADDING))
        #     roi2state.append(nn.ReLU(inplace=True))
        # self.roi2state = nn.Sequential(*roi2state)

        graph = []
        for i in range(self.time_step):
            graph.append(InterNet(self.in_feat_dim))
        self.graph = nn.ModuleList(graph)

        predictor = [nn.Conv2d(self.in_feat_dim * self.time_step, self.in_feat_dim, kernel_size=1), nn.ReLU()]

        for _ in range(C.RPIN.N_EXTRA_PRED_F):
            predictor.append(nn.Conv2d(self.in_feat_dim, self.in_feat_dim,
                                       kernel_size=C.RPIN.EXTRA_F_KERNEL, stride=1, padding=C.RPIN.EXTRA_F_PADDING))
            predictor.append(nn.ReLU(inplace=True))
        self.predictor = nn.Sequential(*predictor)

        self.decoder_output = 3
        # self.bbox_decoder = nn.Linear(self.in_feat_dim * pool_size * pool_size, self.decoder_output)
        self.center3d_world = nn.Linear(self.in_feat_dim * pool_size * pool_size, self.decoder_output)

        # if C.RPIN.MASK_LOSS_WEIGHT > 0:
        #     self.mask_decoder = nn.Sequential(
        #         nn.Linear(self.in_feat_dim * pool_size * pool_size, self.in_feat_dim),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(self.in_feat_dim, self.mask_size * self.mask_size),
        #         nn.Sigmoid(),
        #     )

        # self.center3d_2d_offset_decoder_output = 2
        # if C.RPIN.CENTER3D_2D_OFFSET_LOSS_WEIGHT > 0:
        #     self.center3d_2d_offset_decoder = nn.Linear(self.in_feat_dim * pool_size * pool_size, self.center3d_2d_offset_decoder_output)

        # self.center3d_2d_depth_decoder_output = 1
        # if C.RPIN.CENTER3D_2D_DEPTH_LOSS_WEIGHT > 0:
        #     self.center3d_2d_depth_decoder = nn.Sequential(
        #         nn.Linear(self.in_feat_dim * pool_size * pool_size, self.in_feat_dim),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(self.in_feat_dim, self.center3d_2d_depth_decoder_output),
        #     )
        #     # self.center3d_2d_depth_decoder = nn.Linear(self.in_feat_dim * pool_size * pool_size, self.center3d_2d_depth_decoder_output)

        # if C.RPIN.SEQ_CLS_LOSS_WEIGHT > 0:
        #     self.seq_feature = nn.Sequential(
        #         nn.Linear(self.in_feat_dim * pool_size * pool_size, self.in_feat_dim * 4),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(self.in_feat_dim * 4, self.in_feat_dim),
        #         nn.ReLU(inplace=True),
        #     )
        #     self.seq_score = nn.Sequential(
        #         nn.Linear(self.in_feat_dim * len(self.picked_state_list), 1),
        #         nn.Sigmoid()
        #     )

    def forward(self, features, pointclouds, edges_self, edges_forward, data_pc_oind_tensor, data_pc_oind_help_tensor, data_pc_find_tensor, data_pc_bind_tensor, num_rollouts=10, g_idx=None, x_t=None, phase='train'):
        # x: (b, t, c, h, w)
        # reshape time to batch dimension
        num_objs = data_pc_oind_help_tensor[1:] - data_pc_oind_help_tensor[:-1] -1
        batch_size = data_pc_find_tensor.shape[0]
        time_step = data_pc_find_tensor.shape[1]-1
        assert self.time_step == time_step
        # of shape (b, t, o, dim)
        x = self.extract_object_feature(features, pointclouds, edges_self, edges_forward, data_pc_oind_tensor, data_pc_oind_help_tensor, data_pc_find_tensor, data_pc_bind_tensor)

        # bbox_rollout = []
        # mask_rollout = []
        # center3d_2d_offset_rollout = []
        # center3d_2d_depth_rollout = []
        # print('x',x.shape)
        center3d_world_rollout = []
        state_list = [x[:, i].unsqueeze(-1).unsqueeze(-1) for i in range(self.time_step)]
        state_list_buffer = [x[:, i].unsqueeze(-1).unsqueeze(-1) for i in range(self.time_step)]
        for i in range(num_rollouts):
            c = [self.graph[j](state_list[j], g_idx) for j in range(self.time_step)]
            all_c = torch.cat(c, 2)
            s = self.predictor(all_c.reshape((-1,) + (all_c.shape[-3:])))
            s = s.reshape((batch_size, self.num_objs) + s.shape[-3:])
            center3d_world = self.center3d_world(s.reshape(batch_size, self.num_objs, -1))
            center3d_world_rollout.append(center3d_world)
            # if C.RPIN.MASK_LOSS_WEIGHT:
            #     mask = self.mask_decoder(s.reshape(batch_size, self.num_objs, -1))
            #     mask_rollout.append(mask)
            # if C.RPIN.CENTER3D_2D_OFFSET_LOSS_WEIGHT:
            #     center3d_2d_offset=self.center3d_2d_offset_decoder(s.reshape(batch_size, self.num_objs, -1))
            # if C.RPIN.CENTER3D_2D_DEPTH_LOSS_WEIGHT:
            #     center3d_2d_depth=self.center3d_2d_depth_decoder(s.reshape(batch_size, self.num_objs, -1))
            # bbox_rollout.append(bbox)
            # center3d_2d_offset_rollout.append(center3d_2d_offset)
            # center3d_2d_depth_rollout.append(center3d_2d_depth)
            state_list = state_list[1:] + [s]
            state_list_buffer.append(s)

        # seq_score = []
        # if C.RPIN.SEQ_CLS_LOSS_WEIGHT > 0:
        #     # (p_l * b, o, feat, psz, psz)
        #     state_list_buffer = torch.cat([state_list_buffer[pid] for pid in self.picked_state_list])
        #     # (p_l, b, o, feat)
        #     seq_feature = self.seq_feature(state_list_buffer.reshape(
        #         len(self.picked_state_list) * batch_size, self.num_objs, -1)
        #     ).reshape(len(self.picked_state_list), batch_size, self.num_objs, -1)
        #     valid_seq = g_idx[:, ::self.num_objs - 1, [2]]
        #     valid_seq = valid_seq[None]
        #     # (p_l, b, feat)
        #     seq_feature = (seq_feature * valid_seq).sum(dim=-2) / valid_seq.sum(dim=-2)
        #     seq_feature = seq_feature.permute(1, 2, 0).reshape(batch_size, -1)
        #     seq_score = self.seq_score(seq_feature).squeeze(1)

        # bbox_rollout = torch.stack(bbox_rollout).permute(1, 0, 2, 3)
        # bbox_rollout = bbox_rollout.reshape(-1, num_rollouts, self.num_objs, self.decoder_output)

        center3d_world_rollout = torch.stack(center3d_world_rollout).permute(1, 0, 2, 3)
        center3d_world_rollout = center3d_world_rollout.reshape(-1, num_rollouts, self.num_objs, self.decoder_output)

        # if len(mask_rollout) > 0:
        #     mask_rollout = torch.stack(mask_rollout).permute(1, 0, 2, 3)
        #     mask_rollout = mask_rollout.reshape(-1, num_rollouts, self.num_objs, self.mask_size, self.mask_size)

        # if len(center3d_2d_offset_rollout) > 0:
        #     center3d_2d_offset_rollout = torch.stack(center3d_2d_offset_rollout).permute(1, 0, 2, 3)
        #     center3d_2d_offset_rollout = center3d_2d_offset_rollout.reshape(-1, num_rollouts, self.num_objs, self.center3d_2d_offset_decoder_output)

        # if len(center3d_2d_depth_rollout) > 0:
        #     center3d_2d_depth_rollout = torch.stack(center3d_2d_depth_rollout).permute(1, 0, 2, 3)
        #     center3d_2d_depth_rollout = center3d_2d_depth_rollout.reshape(-1, num_rollouts, self.num_objs, self.center3d_2d_depth_decoder_output)  
        # print('center3d_world_rollout',center3d_world_rollout.shape)
        outputs = {
            'center3d_world': center3d_world_rollout,
        }
        return outputs

    def extract_object_feature(self, features, pointclouds, edges_self, edges_forward, data_pc_oind_tensor, data_pc_oind_help_tensor, data_pc_find_tensor, data_pc_bind_tensor):
        # visual feature, comes from RoI Pooling
        num_objs = data_pc_oind_help_tensor[1:] - data_pc_oind_help_tensor[:-1] -1
        batch_size = data_pc_find_tensor.shape[0]
        time_step = data_pc_find_tensor.shape[1]-1
        
        # # print('features',features)
        # print('features',features.shape)
        # # print('pointclouds',pointclouds)
        # print('pointclouds',len(pointclouds))
        # for i in range(5):
        #     print(f'pointclouds[{i}]',pointclouds[i].shape)
        # # print('edges_self',edges_self)
        # print('edges_self',len(edges_self))
        # for i in range(5):
        #     print(f'edges_self[{i}]',edges_self[i].shape)
        # # print('edges_forward',edges_forward)
        # print('edges_forward',len(edges_forward))
        # for i in range(4):
        #     print(f'edges_forward[{i}]',edges_forward[i].shape)
        # print(" ")

        # print('data_pc_oind_tensor',data_pc_oind_tensor)
        # print('data_pc_oind_help_tensor',data_pc_oind_help_tensor)
        # print('data_pc_find_tensor',data_pc_find_tensor)
        # print('data_pc_bind_tensor',data_pc_bind_tensor)

        # print('num_objs',num_objs)
        
        # print('',)

        # print('data_pc_bind_tensor',data_pc_bind_tensor)
        # for i in range(batch_size):
        #     x = self.backbone(features[:,data_pc_bind_tensor[i]:data_pc_bind_tensor[i+1],:], pointclouds[i:i+3], edges_self, edges_forward)
        x = self.backbone(features, pointclouds, edges_self, edges_forward)
        # print('len(x)',len(x))
        # for i in range(len(x)):
        #     print('x',x[i].shape)
        x_d = x[-1]
        feature_b=[]
        for b in range(batch_size):
            x_b = x_d[:, data_pc_bind_tensor[b]:data_pc_bind_tensor[b+1] ,:]
            feature_f=[]
            for f in range(C.RPIN.INPUT_SIZE):
                x_f = x_b [:, data_pc_find_tensor[b][f]:data_pc_find_tensor[b][f+1] ,:]

                feature_o = []
                for o in range(num_objs[b]):
                    data_pc_oind_tensor_one = data_pc_oind_tensor[f,data_pc_oind_help_tensor[b]:data_pc_oind_help_tensor[b+1]]
                    x_o = x_f[:, data_pc_oind_tensor_one[o]:data_pc_oind_tensor_one[o+1],:]
                    feature_o.append(x_o.mean(1))
                    # print('data_pc_oind_tensor_one',data_pc_oind_tensor_one)
                if len(feature_o) < self.num_objs:
                    feature_o=feature_o + [feature_o[0] for _ in range(C.RPIN.MAX_NUM_OBJS - num_objs[b])]
                feature_o=torch.cat(feature_o,0)
                # print('feature_o.shape',feature_o.shape)
                feature_f.append(feature_o)
            feature_f=torch.stack(feature_f,0)
            feature_b.append(feature_f)
            # print('feature_f.shape',feature_f.shape)
        feature_b=torch.stack(feature_b,0)
        # print('feature_b.shape',feature_b.shape)

        # assert 1==2
        return feature_b
