# Copyright (c) OpenMMLab. All rights reserved.
# modified from
# https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer
from mmcv.cnn.bricks import DropPath
from mmengine.model import ModuleList, Sequential
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.backbones.base_backbone import BaseBackbone
#from mmcls.registry import MODELS

import torch.nn as nn

def build_norm_layer(cfg: dict, num_features: int) -> nn.Module:
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:
            - type (str): The type of normalization layer (e.g., 'BN', 'LN', 'SyncBN').
            - layer args: Additional arguments needed to initialize the normalization layer.
        num_features (int): Number of input channels.

    Returns:
        nn.Module: The created normalization layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict")
    if 'type' not in cfg:
        raise KeyError('The cfg dict must contain the key "type"')

    layer_type = cfg['type']
    layer_args = cfg.get('args', {})  # Optional additional arguments

    if layer_type == 'BN':  # Batch Normalization
        return nn.BatchNorm2d(num_features, **layer_args)
    elif layer_type == 'LN':  # Layer Normalization
        return nn.LayerNorm(num_features, **layer_args)
    elif layer_type == 'SyncBN':  # Synchronized Batch Normalization
        return nn.SyncBatchNorm(num_features, **layer_args)
    elif layer_type == 'GN':  # Group Normalization
        num_groups = layer_args.pop('num_groups', 32)  # Default num_groups
        return nn.GroupNorm(num_groups, num_features, **layer_args)
    else:
        raise ValueError(f"Unsupported normalization type: {layer_type}")


def get_2d_relative_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, grid_size*grid_size]
    """
    pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
    relative_pos = 2 * np.matmul(pos_embed,
                                 pos_embed.transpose()) / pos_embed.shape[1]
    return relative_pos


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed],
                                   axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def xy_pairwise_distance(x, y):
    """Compute pairwise distance of a point cloud.

    Args:
        x: tensor (batch_size, num_points, num_dims)
        y: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        xy_inner = -2 * torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)


def xy_dense_knn_matrix(x, y, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.

    Args:
        x: (batch_size, num_dims, num_points, 1)
        y: (batch_size, num_dims, num_points, 1)
        k: int
        relative_pos:Whether to use relative_pos
    Returns:
        nearest neighbors:
        (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach())
        if relative_pos is not None:
            dist += relative_pos
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(
            0, n_points, device=x.device).repeat(batch_size, k,
                                                 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


class FFN(nn.Module):
    """"out_features = out_features or in_features\n
        hidden_features = hidden_features or in_features"""

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            build_norm_layer(dict(type='BN'), hidden_features),
        )
        self.act = build_activation_layer(act_cfg)
        self.fc2 = Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            build_norm_layer(dict(type='BN'), out_features),
        )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x