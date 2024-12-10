import random
import json
import os
import numpy as np
from utils.pointclouds import PointCloud
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import torchvision.transforms.functional as F
from typing import List, Tuple, Dict


class DetectionDataset6C(Dataset):
    """Dataset class for object detection with 6 channels (RGB+XYZ)"""

    def __init__(
        self,
        data_dirs: List[str],
        coco_paths: List[str],
        coco_types: List[str],
        pcd_additions: List[str],
        transform=None,
        mode: str = "train",
        k: int = 1,
        img_size=768,
    ):
        """
        Initialize the DetectionDataset6C class.

        Args:
            data_dirs (List[str]): List of directories containing data.
            coco_paths (List[str]): List of paths to COCO annotations.
            coco_types (List[str]): List of COCO dataset types ("synthetic" or "real").
            pcd_additions (List[str]): List of PCD file additions.
            transform (optional): Data transformation to apply. Defaults to None.
            mode (str): Dataset mode ("train", "val", or "test"). Defaults to "train".
            k (int): Scalar multiplier for synthetic data. Defaults to 1.
        """
        self.annotations = {}
        self.images_data = {"synthetic": [], "real": []}
        self.seen_classes = []
        self.transform = transform
        self.mode = mode
        self.img_size = (img_size, img_size)

        for data_dir, coco_path, coco_type, pcd_addition in zip(data_dirs, coco_paths, coco_types, pcd_additions):
            self.read_coco(data_dir, coco_path, coco_type, pcd_addition)

        self.to_tensor = ToTensor()

        cutting_p = int(len(self.images_data["real"]) * 0.85)
        random.Random(42).shuffle(self.images_data["real"])

        cutting_p_syn = int(len(self.images_data["synthetic"]) * 0.95)
        random.Random(42).shuffle(self.images_data["synthetic"])

        print(len(self.images_data["real"]), len(self.images_data["synthetic"]))

        if mode == "train":
            print(len(self.images_data["real"][:cutting_p]), len(self.images_data["synthetic"][:cutting_p_syn]))
            self.images_data = self.images_data["real"][:cutting_p] * k + self.images_data["synthetic"][:cutting_p_syn]
        elif mode == "val":
            print(len(self.images_data["real"][cutting_p:]), len(self.images_data["synthetic"][cutting_p_syn:]))
            self.images_data = self.images_data["real"][cutting_p:] + self.images_data["synthetic"][cutting_p_syn:]
        elif mode == "test":
            self.images_data = self.images_data["real"] + self.images_data["synthetic"]
        else:
            print("Mode should be train or val")

    def read_coco(self, data_dir: str, coco_path: str, type_: str, pcd_addition: str):
        """
        Read COCO annotation file and append data to the dataset.

        Args:
            data_dir (str): Directory containing data.
            coco_path (str): Path to COCO annotation file.
            type_ (str): COCO dataset type ("synthetic" or "real").
            pcd_addition (str): PCD file addition.
        """
        with open(coco_path, "r") as f:
            coco = f.read()
        data = json.loads(coco)

        extra = len(self.images_data["real"]) + len(self.images_data["synthetic"])

        self.categories = data["categories"]
        for ann in data["annotations"]:
            bbox = ann["bbox"]
            image_id = ann["image_id"] + extra
            category_id = ann["category_id"]
            class_ = f"{ann['seq']}_{ann['track_id']}"
            if class_ not in self.seen_classes:
                self.seen_classes.append(class_)

            if image_id not in self.annotations:
                self.annotations[image_id] = []
            # Append the current bounding box to the list of bounding boxes for this image
            self.annotations[image_id].append(
                {
                    "category_id": category_id,
                    "bbox": bbox,
                    "track_id": self.seen_classes.index(class_),
                }
            )
        self.images_data[type_].extend(
            [
                [img["id"] + extra, os.path.join(data_dir, img["file_name"]), pcd_addition, type_]
                for img in data["images"]
                if img["id"] + extra in self.annotations.keys()
            ]
        )

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get an item from the dataset.

        Args:
            idx: Index of the item.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]: Tuple containing RGB image, XYZ point cloud,
                and dictionary of labels, bounding boxes, and track labels.
        """
        image_id, image_path, pcd_addition, type_ = self.images_data[idx]
        pcd_path = image_path.split("_color")[0] + pcd_addition + ".pcd"
        # pcd_path = os.path.join(self.img_dir, pcd_path)
        pcd = PointCloud.from_pcd(pcd_path)
        rgb, xyz = pcd.get_points_rgb_xyz_array(reshape=True)

        rgb = Image.fromarray(rgb)
        img_w, img_h = rgb.size

        bboxes = []
        category_ids = []
        track_ids = []
        for annotation in self.annotations[image_id]:
            bbox = annotation["bbox"]
            category_id = annotation["category_id"]
            track_id = annotation["track_id"]
            x1, y1, width, height = bbox
            cx, cy = x1 + width / 2, y1 + height / 2
            bbox = [cx / img_w, cy / img_h, width / img_w, height / img_h]
            bboxes.append(bbox)
            category_ids.append(category_id)
            track_ids.append(track_id)

        labels = torch.zeros((len(bboxes),), dtype=torch.int64)  # this should be category_ids if more than 1 class
        track_labels = torch.tensor(track_ids, dtype=torch.int64)

        # Convert boxes array to torch tensor
        bboxes = torch.tensor(bboxes, dtype=torch.float32)

        if self.mode == "train" and random.random() < 0.5:
            crop = (
                random.uniform(0.3, 0.7),
                random.uniform(0.3, 0.7),
                random.uniform(0.4, 0.9),
                random.uniform(0.4, 0.9),
            )
            rgb, xyz, bboxes, labels, track_labels = crop_image_and_boxes(rgb, xyz, bboxes, labels, track_labels, crop)

        # Since the number of bounding boxes (aka leaves) per image is different, we need to
        # create illegal boxes (with label=-1) so all images have the same number of boxes
        # and we can create batches
        illegal_needed = 33 - len(bboxes)
        illegal_labels = torch.ones((illegal_needed,), dtype=torch.int64) * -1
        illegal_boxes = torch.zeros((illegal_needed, 4), dtype=torch.float32) * -1
        illegal_track_labels = torch.ones((illegal_needed,), dtype=torch.int64) * -1

        if self.transform:
            rgb = self.transform(rgb)
        rgb = self.to_tensor(rgb)

        if self.mode == "train":
            # xyz = xyz + np.random.normal(0, 0.001, xyz.shape).astype(np.float32)
            t_noise = np.array(
                [
                    random.uniform(-0.005, 0.005),
                    random.uniform(-0.005, 0.005),
                    random.uniform(-0.005, 0.005),
                ],
                dtype=np.float32,
            )
            xyz = xyz + t_noise

        xyz = np.nan_to_num(xyz, nan=0.0)

        if type_ == "synthetic":
            xyz[:, :, 0] = channel_to_heatmap(xyz[:, :, 0], min=-0.33775, max=0.34639)
            xyz[:, :, 1] = channel_to_heatmap(xyz[:, :, 1], min=-0.34417, max=0.35452)
            xyz[:, :, 2] = channel_to_heatmap(xyz[:, :, 2], min=-0.09238, max=2.26342)
        else:
            xyz[:, :, 0] = channel_to_heatmap(xyz[:, :, 0], min=-0.4, max=0.8)
            xyz[:, :, 1] = channel_to_heatmap(np.absolute(xyz[:, :, 1]), min=0, max=0.8)
            xyz[:, :, 2] = channel_to_heatmap(xyz[:, :, 2], min=0.2, max=1.5)

        xyz = self.to_tensor(xyz)

        # if self.mode == "train":
        rgb = F.resize(rgb, self.img_size, antialias=True)
        xyz = F.resize(xyz, self.img_size, antialias=True)

        return (
            rgb,
            xyz,
            {
                "labels": torch.cat((labels, illegal_labels)),
                "boxes": torch.cat((bboxes, illegal_boxes), axis=0),
                "track_labels": torch.cat((track_labels, illegal_track_labels)),
            },
        )

    def save_by_idx(self, idx):
        rgb, xyz, _ = self.__getitem__(idx)
        save_image(rgb, "rgb.png")
        save_image(xyz, "xyz.png")


def channel_to_heatmap(depth_img, min, max):
    """Generate a heatmap out of a depth image. A distance threshold can be set to
    increase the resolution of the heatmap depth-wise.

    Args:
        depth_img (np.array): depthmap (w, h).
        dist_threshold (int, optional): distance clipping threshold. Defaults to 10000.

    Returns:
        np.array: heatmap as image (w, h, 3) format.
    """
    indices_0 = np.where((depth_img > max) | (depth_img < min) | (depth_img == 0))
    depth_img[indices_0] = 1e5
    depth_img[indices_0] = np.min(depth_img)
    depth_img = (depth_img - min) / (max - min)
    # depth_colormap = (depth_img * 255).astype(np.uint8)
    depth_img[indices_0] = 0
    return depth_img


def crop_image_and_boxes(image, xyz, boxes, labels, track_labels, crop):
    """
    image: a PIL.Image object
    boxes: a tensor of bounding boxes in the format [[x_center, y_center, width, height], ...]
    labels: a tensor of labels
    crop: a tuple of (x_center, y_center, width, height) normalized with respect to image dimensions
    """
    # Convert boxes to pixel coordinates
    img_width, img_height = image.size
    boxes = boxes * torch.tensor([img_width, img_height, img_width, img_height])

    # Convert from (x_center, y_center, width, height) to (xmin, ymin, xmax, ymax)
    boxes = torch.stack(
        [
            boxes[:, 0] - boxes[:, 2] / 2,
            boxes[:, 1] - boxes[:, 3] / 2,
            boxes[:, 0] + boxes[:, 2] / 2,
            boxes[:, 1] + boxes[:, 3] / 2,
        ],
        dim=1,
    )

    # Convert the crop region from normalized (x_center, y_center, width, height) to pixel (left, upper, right, lower)
    x_center, y_center, width, height = crop
    x_center, y_center, width, height = (
        x_center * img_width,
        y_center * img_height,
        width * img_width,
        height * img_height,
    )
    crop = (
        int(max(0, x_center - width / 2)),
        int(max(0, y_center - height / 2)),
        int(min(img_width, x_center + width / 2)),
        int(min(img_height, y_center + height / 2)),
    )

    # Crop the image
    cropped_image = image.crop(crop)
    if xyz is not None:
        cropped_xyz = xyz[crop[1] : crop[3], crop[0] : crop[2], :]
    else:
        cropped_xyz = None
    # Adjust bounding boxes
    left, upper, right, lower = crop
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= upper

    # Clip the bounding boxes' coordinates to be in [0, width/height of the cropped image]
    boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], min=0, max=cropped_image.width)
    boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], min=0, max=cropped_image.height)

    # Filter out the boxes that are completely outside the cropped region
    mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[mask]
    labels = labels[mask]
    track_labels = track_labels[mask]

    # Convert back to (x_center, y_center, width, height) format
    boxes = torch.stack(
        [
            (boxes[:, 0] + boxes[:, 2]) / 2,
            (boxes[:, 1] + boxes[:, 3]) / 2,
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1],
        ],
        dim=1,
    )

    # Normalize the bounding boxes with respect to the new dimensions
    new_width, new_height = cropped_image.size
    boxes = boxes / torch.tensor([new_width, new_height, new_width, new_height])

    return cropped_image, cropped_xyz, boxes, labels, track_labels
