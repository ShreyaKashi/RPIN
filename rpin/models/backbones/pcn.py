#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from rpin.models.backbones.pcn_layers import PointConv, PointConvStridePE, PointConvTransposePE, PointTransformerLayer, PCFLayer, Linear_BN


def get_default_configs(cfg, num_level=5, base_dim=64):
    '''
    Get the default configurations for the model, returns the config EasyDict populated with the default values
    '''
    # Number of downsampling stages
    cfg.num_level = num_level
    # The dimensionality of the first stage
    cfg.base_dim = base_dim
    # Feature dimensionality for each stage
    if 'feat_dim' not in cfg.keys():
        cfg.feat_dim = [base_dim * (i + 1) for i in range(cfg.num_level + 1)]
    # Whether to use the viewpoint-invariant coordinate transforms
    # (Xingyi Li et al. Improving the Robustness of Point Convolution on K-Nearest Neighbor Neighborhoods with a Viewpoint-Invariant Coordinate Transform. WACV 2023)
    if 'USE_VI' not in cfg.keys():
        # cfg.USE_VI = True
        cfg.USE_VI = False #change by mingqi
    # Whether to concatenate positional encoding into features for VI_PointConv, improves performance at the cost of more GPU memory
    if 'USE_PE' not in cfg.keys():
        cfg.USE_PE = False
    # Transformer type: 'PCF' or anything else (PointTransformer)
    if 'transformer_type' not in cfg.keys():
        cfg.transformer_type = 'PCF'
    # Attention type: 'subtraction': subtractive attention, anything else:
    # QK-style attention
    if 'attention_type' not in cfg.keys():
        cfg.attention_type = 'subtraction'
    # Whether to use a layer norm in the guidance computation
    if 'layer_norm_guidance' not in cfg.keys():
        cfg.layer_norm_guidance = False
    # Whether to use drop path
    if 'drop_path_rate' not in cfg.keys():
        cfg.drop_path_rate = 0.
    # Whether to use batch normalization
    if 'BATCH_NORM' not in cfg.keys():
        cfg.BATCH_NORM = True
    # Dropout rate
    if 'dropout_rate' not in cfg.keys():
        cfg.dropout_rate = 0.
    # Whether to time the individual components of the PointConv layers
    if 'TIME' not in cfg.keys():
        cfg.TIME = False
    # Whether to use coordinates as features too
    if 'USE_XYZ' not in cfg.keys():
        cfg.USE_XYZ = True
    # The dimensionality of the point cloud coordinates
    if 'point_dim' not in cfg.keys():
        cfg.point_dim = 3
    # Transformer_type: this can be either PCF or PointTransformer
    # for now, if it's not set to PCF then we will use PointTransformer
    if 'transformer_type' not in cfg.keys():
        cfg.transformer_type = 'PCF'
    # mid_dim_back: c_mid in the decoder (see the paper)
    if 'mid_dim_back' not in cfg.keys():
        cfg.mid_dim_back = 1
    # Whether to use PointConvs at the original input resolution, they are
    # helpful for performance esp. on 10cm and 5cm, but uses more memory esp.
    # on 2cm
    if 'use_level_1' not in cfg.keys():
        cfg.use_level_1 = True
    # Whether to use the CUDA kernels for PointConv and PointConvFormer which fuses
    # gather operations and save a few milliseconds and some memory
    # Right now, using CUDA kernels to train mysteriously reduces training
    # accuracy, hence it is only recommmended to be used during testing time
    if 'USE_CUDA_KERNEL' not in cfg.keys():
        cfg.USE_CUDA_KERNEL = False
    return cfg


class PCF_Backbone(nn.Module):
    '''
    The backbone-only part of PCF
    input is from the data loader (features, point cloud coordinates, edges_self, edges_forward, norms)

    The entire model, including the backbone and the segmentation decoder
    input is from the data loader (features, point cloud coordinates, edges_self, edges_forward, edges_propagate, norms)
    Input:
        features: 1 x num_points x num_channels: features of all point clouds in the packed format
        pointclouds:  a list of [1 x num_points[i] x 3]: coordinates of all point clouds at different subsampling levels in the packed format
        edges_self: a list of [1 x num_points[i] x K_self]: Neighbor indices between points at the same subsampling level
        edges_forward: a list of [1 x num_points[i+1] x K_forward]: Neighbor indices between points at one subsampling level and points at the next subsampling level (more sparse)
    Output:
        feat_list: a list of [1 x num_points[i] x out_channels[i]], output
                   features at each subsampling level
    '''
    def __init__(self, cfg, input_feat_dim=3):
        super(PCF_Backbone, self).__init__()

        self.cfg = cfg
        self.total_level = cfg.num_level
        self.guided_level = cfg.guided_level

        self.input_feat_dim = input_feat_dim + 3 if cfg.USE_XYZ else input_feat_dim

        self.relu = torch.nn.ReLU(inplace=True)

        if cfg.USE_VI is True:
            weightnet_input_dim = cfg.point_dim + 9
        else:
            weightnet_input_dim = cfg.point_dim
        weightnet = [weightnet_input_dim, cfg.mid_dim[0]]  # 2 hidden layer
        weightnet_start = [weightnet_input_dim, cfg.mid_dim[0]]

        if cfg.use_level_1:
            self.selfpointconv = PointConv(
                self.input_feat_dim, cfg.base_dim, cfg, weightnet_start)
            self.selfpointconv_res1 = PointConvStridePE(
                cfg.base_dim, cfg.base_dim, cfg, weightnet_start)
            self.selfpointconv_res2 = PointConvStridePE(
                cfg.base_dim, cfg.base_dim, cfg, weightnet_start)
        else:
            self.selfmlp = Linear_BN(
                self.input_feat_dim, cfg.base_dim, bn_ver='1d')

        self.pointconv = nn.ModuleList()
        self.pointconv_res = nn.ModuleList()
        # print('self.total_level',self.total_level)
        for i in range(1, self.total_level):
            # print('i',i)
            in_ch = cfg.feat_dim[i - 1]
            out_ch = cfg.feat_dim[i]
            weightnet = [weightnet_input_dim, cfg.mid_dim[i]]

            if i <= self.guided_level:
                self.pointconv.append(
                    PointConvStridePE(
                        in_ch, out_ch, cfg, weightnet))
            else:
                if self.cfg.transformer_type == 'PCF':
                    self.pointconv.append(
                        PCFLayer(
                            in_ch,
                            out_ch,
                            cfg,
                            weightnet,
                            cfg.num_heads))
                else:
                    self.pointconv.append(
                        PointTransformerLayer(
                            in_ch, out_ch, cfg.num_heads))

            if self.cfg.resblocks[i] == 0:
                self.pointconv_res.append(nn.ModuleList([]))
            else:
                res_blocks = nn.ModuleList()
                for _ in range(self.cfg.resblocks[i]):
                    if i <= self.guided_level:
                        res_blocks.append(
                            PointConvStridePE(
                                out_ch, out_ch, cfg, weightnet))
                    else:
                        if self.cfg.transformer_type == 'PCF':
                            res_blocks.append(
                                PCFLayer(
                                    out_ch,
                                    out_ch,
                                    cfg,
                                    weightnet,
                                    cfg.num_heads))
                        else:
                            res_blocks.append(
                                PointTransformerLayer(
                                    out_ch, out_ch, cfg.num_heads))
                self.pointconv_res.append(res_blocks)

    def forward(self, features, pointclouds, edges_self, edges_forward, norms=None):
        # encode pointwise info
        pointwise_feat = torch.cat(
            [features, pointclouds[0]], -1) if self.cfg.USE_XYZ else features

        # level 1 conv, this helps performance significantly on 5cm/10cm inputs
        # but have relatively small use on 2cm
        if self.cfg.use_level_1:
            pointwise_feat, vi_features = self.selfpointconv(
                pointclouds[0], pointwise_feat, edges_self[0], norms)
            pointwise_feat, _ = self.selfpointconv_res1(
                pointclouds[0], pointwise_feat, edges_self[0], norms, vi_features=vi_features)
            pointwise_feat, _ = self.selfpointconv_res2(
                pointclouds[0], pointwise_feat, edges_self[0], norms, vi_features=vi_features)
        else:
            # if don't use level 1 convs, then just simply do a linear layer to
            # increase the feature dimensionality
            pointwise_feat = F.relu(self.selfmlp(pointwise_feat), inplace=True)

        feat_list = [pointwise_feat]
        for i, pointconv in enumerate(self.pointconv):
            if self.cfg.transformer_type != 'PCF':
                sparse_feat = pointconv(
                    pointclouds[i], feat_list[-1], edges_forward[i], pointclouds[i + 1])
            else:
                sparse_feat, _ = pointconv(
                    pointclouds[i], feat_list[-1], edges_forward[i], norms, pointclouds[i + 1], norms)
            # print(sparse_feat.shape)
            # There is the need to recompute VI features from the neighbors at this level rather than from the previous level, hence need
            # to recompute VI features in the first residual block
            vi_features = None
            for res_block in self.pointconv_res[i]:
                if self.cfg.transformer_type != 'PCF':
                    sparse_feat = res_block(
                        pointclouds[i + 1], sparse_feat, edges_self[i + 1])
                else:
                    if vi_features is not None:
                        sparse_feat, _ = res_block(
                            pointclouds[i + 1], sparse_feat, edges_self[i + 1], norms, vi_features=vi_features)
                    else:
                        sparse_feat, vi_features = res_block(
                            pointclouds[i + 1], sparse_feat, edges_self[i + 1], norms)

            feat_list.append(sparse_feat)

        return feat_list


def PCF_Tiny(input_grid_size, base_dim=64):
    ''' A tiny PCF model. 
    Input:
        input_grid_size: the voxel size at the finest resolution, e.g. 0.02 for 2cm, 0.05 for 5cm, 0.1 for 10cm
    Output:
        PCF_Backbone: A PCF backbone model
        cfg: The configuration options initialized for the PCF backbone model
    '''
    cfg = EasyDict()
    cfg = get_default_configs(cfg, num_level=5, base_dim=base_dim)
    cfg.guided_level = 0
    cfg.num_heads = 1
    cfg.resblocks = [0, 1, 1, 1, 1]
    cfg.mid_dim = [4, 4, 4, 4, 4]
    cfg.grid_size = [
        input_grid_size,
        input_grid_size * 2,
        input_grid_size * 4,
        input_grid_size * 8,
        input_grid_size * 16]
    return PCF_Backbone(cfg), cfg

# A small model


def PCF_Small(input_grid_size, base_dim=64):
    ''' A small PCF model. 
    Input:
        input_grid_size: the voxel size at the finest resolution, e.g. 0.02 for 2cm, 0.05 for 5cm, 0.1 for 10cm
    Output:
        PCF_Backbone: A PCF backbone model
        cfg: The configuration options initialized for the PCF backbone model
    '''
    cfg = EasyDict()
    cfg = get_default_configs(cfg, num_level=5, base_dim=base_dim)
    cfg.guided_level = 0
    cfg.num_heads = 8
    cfg.resblocks = [0, 2, 2, 2, 2]
    cfg.mid_dim = [4, 4, 4, 4, 4]
    cfg.grid_size = [
        input_grid_size,
        input_grid_size * 2,
        input_grid_size * 4,
        input_grid_size * 8,
        input_grid_size * 16]
    return PCF_Backbone(cfg), cfg

# A normal sized model


def PCF_Normal(input_grid_size, base_dim=64):
    ''' A normal PCF model. 
    Input:
        input_grid_size: the voxel size at the finest resolution, e.g. 0.02 for 2cm, 0.05 for 5cm, 0.1 for 10cm
    Output:
        PCF_Backbone: A PCF backbone model
        cfg: The configuration options initialized for the PCF backbone model
    '''
    cfg = EasyDict()
    cfg = get_default_configs(cfg, num_level=5, base_dim=base_dim)
    cfg.guided_level = 0
    cfg.num_heads = 8
    cfg.resblocks = [0, 2, 4, 6, 6]
    cfg.grid_size = [
        input_grid_size,
        input_grid_size * 2,
        input_grid_size * 4,
        input_grid_size * 8,
        input_grid_size * 16]
    cfg.mid_dim = [16, 16, 16, 16, 16]
    return PCF_Backbone(cfg), cfg


def PCF_Large(input_grid_size, base_dim=64):
    ''' A large PCF model. 
    Input:
        input_grid_size: the voxel size at the finest resolution, e.g. 0.02 for 2cm, 0.05 for 5cm, 0.1 for 10cm
    Output:
        PCF_Backbone: A PCF backbone model
        cfg: The configuration options initialized for the PCF backbone model
    '''
    cfg = EasyDict()
    cfg = get_default_configs(cfg, num_level=6, base_dim=base_dim)
    cfg.guided_level = 0
    cfg.num_heads = 8
    cfg.resblocks = [0, 2, 4, 6, 6, 2]
    cfg.grid_size = [
        input_grid_size,
        input_grid_size * 2.5,
        input_grid_size * 5,
        input_grid_size * 10,
        input_grid_size * 20,
        input_grid_size * 40]
    cfg.mid_dim = [16, 16, 16, 16, 16, 16]
    return PCF_Backbone(cfg), cfg