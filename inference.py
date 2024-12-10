import numpy as np
import torch
from torchvision.transforms import ToTensor

from models.mot_detr import MOTDETR
from datasets.dataset import channel_to_heatmap
from utils.pointclouds import PointCloud

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Load data from PCD file
pcd = PointCloud.from_pcd("sample_data/1680274967710036277_b.pcd")
viewpoint = pcd.viewpoint
pcd.transform_to_origin()
rgb, xyz = pcd.get_points_rgb_xyz_array(reshape=True)
xyz[:, :, 0] = channel_to_heatmap(xyz[:, :, 0], min=-0.4, max=0.8)
xyz[:, :, 1] = channel_to_heatmap(np.absolute(xyz[:, :, 1]), min=0, max=0.8)
xyz[:, :, 2] = channel_to_heatmap(xyz[:, :, 2], min=0.2, max=1.5)
xyz = np.nan_to_num(xyz, nan=0.0)
to_tensor = ToTensor()

# Load model from weights
detector = MOTDETR.load_from_checkpoint("best_resnet34.ckpt")

# Inference
boxes, top_class, scores, _, features = detector.detect(
    to_tensor(rgb), to_tensor(xyz), device=device, nms_threshold=0.4
)
print(boxes)
