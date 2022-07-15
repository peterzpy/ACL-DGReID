# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from fastreid.config import configurable
from fastreid.layers import *
from fastreid.layers import pooling, any_softmax
from fastreid.layers.weight_init import weights_init_kaiming
from .build import REID_HEADS_REGISTRY


@REID_HEADS_REGISTRY.register()
class EmbeddingHead(nn.Module):
    """
    EmbeddingHead perform all feature aggregation in an embedding task, such as reid, image retrieval
    and face recognition

    It typically contains logic to

    1. feature aggregation via global average pooling and generalized mean pooling
    2. (optional) batchnorm, dimension reduction and etc.
    2. (in training only) margin-based softmax logits computation
    """

    @configurable
    def __init__(
            self,
            *,
            feat_dim,
            embedding_dim,
            num_classes1,
            num_classes2,
            num_classes3,
            neck_feat,
            pool_type,
            cls_type,
            scale,
            margin,
            with_bnneck,
            norm_type
    ):
        """
        NOTE: this interface is experimental.

        Args:
            feat_dim:
            embedding_dim:
            num_classes:
            neck_feat:
            pool_type:
            cls_type:
            scale:
            margin:
            with_bnneck:
            norm_type:
        """
        super().__init__()

        # Pooling layer
        assert hasattr(pooling, pool_type), "Expected pool types are {}, " \
                                            "but got {}".format(pooling.__all__, pool_type)
        self.pool_layer = getattr(pooling, pool_type)()

        self.neck_feat = neck_feat

        neck = []
        if embedding_dim > 0:
            neck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            neck.append(get_norm(norm_type, feat_dim, bias_freeze=True))

        self.bottleneck = nn.Sequential(*neck)

        # Classification head
        assert hasattr(any_softmax, cls_type), "Expected cls types are {}, " \
                                               "but got {}".format(any_softmax.__all__, cls_type)
        self.weight1 = nn.Parameter(torch.Tensor(num_classes1, feat_dim))
        self.weight2 = nn.Parameter(torch.Tensor(num_classes2, feat_dim))
        self.weight3 = nn.Parameter(torch.Tensor(num_classes3, feat_dim))
        self.cls_layer1 = getattr(any_softmax, cls_type)(num_classes1, scale, margin)
        self.cls_layer2 = getattr(any_softmax, cls_type)(num_classes2, scale, margin)
        self.cls_layer3 = getattr(any_softmax, cls_type)(num_classes3, scale, margin)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.bottleneck.apply(weights_init_kaiming)
        nn.init.normal_(self.weight1, std=0.01)
        nn.init.normal_(self.weight2, std=0.01)
        nn.init.normal_(self.weight3, std=0.01)

    @classmethod
    def from_config(cls, cfg):
        # fmt: off
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes1   = cfg.MODEL.HEADS.NUM_CLASSES1
        num_classes2   = cfg.MODEL.HEADS.NUM_CLASSES2
        num_classes3   = cfg.MODEL.HEADS.NUM_CLASSES3
        neck_feat     = cfg.MODEL.HEADS.NECK_FEAT
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER
        cls_type      = cfg.MODEL.HEADS.CLS_LAYER
        scale         = cfg.MODEL.HEADS.SCALE
        margin        = cfg.MODEL.HEADS.MARGIN
        with_bnneck   = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type     = cfg.MODEL.HEADS.NORM
        # fmt: on
        return {
            'feat_dim': feat_dim,
            'embedding_dim': embedding_dim,
            'num_classes1': num_classes1,
            'num_classes2': num_classes2,
            'num_classes3': num_classes3,
            'neck_feat': neck_feat,
            'pool_type': pool_type,
            'cls_type': cls_type,
            'scale': scale,
            'margin': margin,
            'with_bnneck': with_bnneck,
            'norm_type': norm_type
        }

    #CHANGE Add reduction version
    def forward(self, features, targets=None, reduction=False, paths=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        pool_feat = self.pool_layer(features)

        if paths is not None:
            pool_feat = torch.cat([pool_feat, paths.unsqueeze(-1).unsqueeze(-1)], 1)

        neck_feat = self.bottleneck(pool_feat)
        neck_feat = neck_feat[..., 0, 0]
        
        if not reduction:
            # Evaluation
            # fmt: off
            if not self.training: return neck_feat
            # fmt: on

            # Training
            if self.cls_layer1.__class__.__name__ == 'Linear':
                logits1 = F.linear(neck_feat, self.weight1)
                logits2 = F.linear(neck_feat, self.weight2)
                logits3 = F.linear(neck_feat, self.weight3)
                
            else:
                logits1 = F.linear(F.normalize(neck_feat), F.normalize(self.weight1))
                logits2 = F.linear(F.normalize(neck_feat), F.normalize(self.weight2))
                logits3 = F.linear(F.normalize(neck_feat), F.normalize(self.weight3))

            # Pass logits.clone() into cls_layer, because there is in-place operations
            cls_outputs1 = self.cls_layer1(logits1.clone(), targets)
            cls_outputs2 = self.cls_layer2(logits2.clone(), targets)
            cls_outputs3 = self.cls_layer3(logits3.clone(), targets)

            # fmt: off
            if self.neck_feat == 'before':  feat = pool_feat[..., 0, 0]
            elif self.neck_feat == 'after': feat = neck_feat
            else:                           raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")
            # fmt: on

            return {
                "cls_outputs1": cls_outputs1,
                "cls_outputs2": cls_outputs2,
                "cls_outputs3": cls_outputs3,
                "pred_class_logits1": logits1.mul(self.cls_layer1.s),
                "pred_class_logits2": logits2.mul(self.cls_layer2.s),
                "pred_class_logits3": logits3.mul(self.cls_layer3.s),
                "features": feat,
            }
        else: 
            # fmt: off
            if self.neck_feat == 'before':  feat = pool_feat[..., 0, 0]
            elif self.neck_feat == 'after': feat = neck_feat
            else:                           raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")
            # fmt: on
            return {
                "features": feat,
            }
