import math
import torch
from torch import nn

import torchvision
from torchvision.ops import FrozenBatchNorm2d

import torch.utils.data

from .transformer import Transformer


class PositionEmbeddingSine(nn.Module):
    """Generate position embeddings like DETR"""

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class DETRMOT(nn.Module):
    """MOT-DETR netowrk."""

    def __init__(
        self,
        num_classes: int,
        num_track_classes: int,
        hidden_dim: int = 128,
        nheads: int = 4,
        num_queries: int = 20,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
    ):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone_cnn = nn.Sequential(
            *list(torchvision.models.resnet34(weights="DEFAULT", norm_layer=FrozenBatchNorm2d).children())[:-2]
        )
        self.backbone_pc = nn.Sequential(
            *list(torchvision.models.resnet34(weights="DEFAULT", norm_layer=FrozenBatchNorm2d).children())[:-2]
        )

        # create conversion layer
        self.conv = nn.Conv2d(512, hidden_dim, 1)
        self.conv_pc = nn.Conv2d(512, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = Transformer(
            2 * hidden_dim,
            nheads,
            num_encoder_layers,
            num_decoder_layers,
            normalize_before=True,
        )

        # prediction heads, one extra class for predicting non-empty slots
        self.linear_class = nn.Linear(2 * hidden_dim, num_classes + 1)

        self.linear_bbox = nn.Sequential(
            nn.Linear(2 * hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )
        self.linear_track_feats = nn.Sequential(
            nn.Linear(2 * hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.linear_track_class = nn.Linear(128, num_track_classes)

        # output positional encodings (object queries)
        self.query_pos = nn.Embedding(num_queries, 2 * hidden_dim)

        # spatial positional encodings
        self.position_embedding = PositionEmbeddingSine((2 * hidden_dim) // 2)

        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

    def forward(self, rgb, xyz):
        batch_size = rgb.shape[0]

        # propagate inputs through ResNet-50 up to avg-pool layer
        x_cnn = self.backbone_cnn(rgb)

        # convert from 512 to 128 feature planes for the transformer
        h_cnn = self.conv(x_cnn)
        H, W = x_cnn.shape[-2:]

        x_pc = self.backbone_pc(xyz)
        h_pc = self.conv_pc(x_pc)

        h = torch.cat((h_cnn, h_pc), dim=1)
        # print(h_cnn.shape, h_pc.shape, h.shape)
        mask = h.new_zeros((batch_size, H, W), dtype=torch.bool)

        pos = self.position_embedding(h, mask)

        # propagate through the transformer
        h = self.transformer(h, self.query_pos.weight, pos)[0]

        # finally project transformer outputs to class labels and bounding boxes
        pred_logits = self.linear_class(h)
        track_feats = self.linear_track_feats(h)
        pred_tracks = self.linear_track_class(track_feats)
        return {
            "pred_logits": pred_logits,
            "pred_boxes": self.linear_bbox(h).sigmoid(),
            "pred_tracks": pred_tracks,
            "features": track_feats,
        }
