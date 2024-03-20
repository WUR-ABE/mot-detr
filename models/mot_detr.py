import random
from typing import List

# PyTorch
import torch
import torch.nn.functional as F
import torch.optim as optim

# Torchvision
import torchvision
import torchvision.transforms.functional as torchvisionF

# Pytorch lightning
import pytorch_lightning as pl

from utils.utils import generalized_box_iou, box_cxcywh_to_xyxy, mean_average_precision, accuracy
from models.model_simple import DETRMOT
from utils.matcher import HungarianMatcher, get_src_permutation_idx

torch.set_float32_matmul_precision("high")

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class MOTDETR(pl.LightningModule):
    """MOT-DETR: single-shot detection and tracking with transformers. The network is trained using
    Pytorch Lightning."""

    def __init__(
        self,
        hidden_dim: int,
        lr: float,
        weight_decay: float = 1e-4,
        num_classes: int = 1,
        num_track_classes: int = 1,
        nheads: int = 4,
        num_queries: int = 20,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        batch_size: int = 8,
        scheduler_steps: List[int] = [30000, 50000, 70000, 90000],
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = DETRMOT(
            num_classes=num_classes,
            num_track_classes=num_track_classes,
            hidden_dim=hidden_dim,
            nheads=nheads,
            num_queries=num_queries,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )

        self.matcher = HungarianMatcher()
        self.num_classes = num_classes
        self.preds = []
        self.GTs = []

        self.sizes = [
            (512, 512),
            (640, 640),
            (540, 960),
            (378, 672),
            (432, 768),
            (486, 864),
        ]

    def configure_optimizers(self):
        params_cnn = [
            param for name, param in self.model.named_parameters() if "backbone_cnn" in name or "backbone_pc" in name
        ]
        params_rest = [
            param
            for name, param in self.model.named_parameters()
            if "backbone_cnn" not in name and "backbone_pc" not in name
        ]

        optimizer = torch.optim.AdamW(
            [{"params": params_rest}, {"params": params_cnn, "lr": self.hparams.lr * 0.1}],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = {
            "scheduler": optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.hparams.scheduler_steps, gamma=0.5
            ),  # StepLR(optimizer, step_size=15000, gamma=0.5),
            "interval": "step",
        }

        return [optimizer], [scheduler]

    def calc_loss(self, batch, mode="train"):
        rgb, xyz, targets = batch
        target_size = random.choice(self.sizes)
        rgb = torchvisionF.resize(rgb, target_size, antialias=True)
        xyz = torchvisionF.resize(xyz, target_size, antialias=True)

        rgb = rgb.to(device)
        xyz = xyz.to(device)

        # Remove illegal targets
        new_targets = []
        for i in range(targets["labels"].shape[0]):
            labels = targets["labels"][i]
            boxes = targets["boxes"][i]
            track_labels = targets["track_labels"][i]
            new_targets.append(
                {
                    "labels": labels[labels != -1].to(device),
                    "boxes": boxes[labels != -1].to(device),
                    "track_labels": track_labels[track_labels != -1].to(device),
                }
            )

        outputs = self.model(rgb, xyz)

        num_boxes = sum(len(t["labels"]) for t in new_targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        src_logits = outputs["pred_logits"]
        indices = self.matcher(outputs, new_targets)  # Run matcher
        idx = get_src_permutation_idx(indices)

        # Loss class
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(new_targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 1, dtype=torch.int64, device=device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)

        # loss boxes L1 and GIOU
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(new_targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou.sum() / num_boxes

        # Loss track class
        pred_tracks = outputs["pred_tracks"][idx]
        target_tracks = torch.cat([t["track_labels"][J] for t, (_, J) in zip(new_targets, indices)])
        track_loss = F.cross_entropy(pred_tracks, target_tracks)

        det_loss = loss_ce + 5 * loss_bbox + 2 * loss_giou
        loss = (
            torch.exp(-self.model.s_det) * det_loss
            + torch.exp(-self.model.s_id) * track_loss
            + (self.model.s_det + self.model.s_id)
        )
        loss *= 0.5

        # Logging loss
        self.log("weight_det", self.model.s_det)
        self.log("weight_id", self.model.s_id)
        self.log(mode + "_loss", loss)
        self.log(mode + "_loss_ce", loss_ce)
        self.log(mode + "_loss_bbox", loss_bbox)
        self.log(mode + "_loss_giou", loss_giou)
        self.log(mode + "_loss_ce_track", track_loss)

        # Logging accuracy
        with torch.no_grad():
            top1, top5 = accuracy(pred_tracks, target_tracks, topk=(1, 5))
            self.log(mode + "_acc_top1", top1)
            self.log(mode + "_acc_top5", top5)

        return loss

    def training_step(self, batch, batch_idx):
        return self.calc_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        rgb, xyz, targets = batch
        rgb = rgb.to(device)
        xyz = xyz.to(device)

        new_targets = []
        new_targets_tensors = []
        for idx in range(targets["labels"].shape[0]):
            labels = targets["labels"][idx]
            boxes = targets["boxes"][idx]
            track_labels = targets["track_labels"][idx]
            new_targets.append(
                {
                    "labels": labels[labels != -1].cpu().detach().numpy(),
                    "boxes": boxes[labels != -1].cpu().detach().numpy(),
                }
            )

            new_targets_tensors.append(
                {
                    "labels": labels[labels != -1].to(device),
                    "boxes": boxes[labels != -1].to(device),
                    "track_labels": track_labels[track_labels != -1].to(device),
                }
            )

        for j in range(rgb.shape[0]):
            for k in range(new_targets[j]["labels"].shape[0]):
                label_info = []
                label_info.append(batch_idx * self.hparams.batch_size + j)  # image index
                label_info.append(new_targets[j]["labels"][k])  # class label
                label_info.append(1)  # class label
                label_info.extend(new_targets[j]["boxes"][k].tolist())  # bounding box coordinates
                self.GTs.append(label_info)

        outputs = self.model(rgb, xyz)

        indices = self.matcher(outputs, new_targets_tensors)  # Run matcher
        idx = get_src_permutation_idx(indices)
        pred_tracks = outputs["pred_tracks"][idx]
        target_tracks = torch.cat([t["track_labels"][J] for t, (_, J) in zip(new_targets_tensors, indices)])
        top1, top5 = accuracy(pred_tracks, target_tracks, topk=(1, 5))
        self.log("val" + "_acc_top1", top1)
        self.log("val" + "_acc_top5", top5)

        outputs["pred_logits"] = outputs["pred_logits"].cpu().float()
        outputs["pred_boxes"] = outputs["pred_boxes"].cpu().float()

        prob = F.softmax(outputs["pred_logits"][0], dim=1)
        top_p, top_class = prob.topk(1, dim=1)
        # top_class = torch.where(top_p > 0.7, top_class, self.num_classes)

        boxes = outputs["pred_boxes"][0][top_class.squeeze() != self.num_classes]
        scores = top_p[top_class != self.num_classes]
        top_class = top_class[top_class != self.num_classes]

        sel_boxes_idx = torchvision.ops.nms(boxes=box_cxcywh_to_xyxy(boxes), scores=scores, iou_threshold=0.5)

        boxes = boxes[sel_boxes_idx].cpu().detach().numpy()
        scores = scores[sel_boxes_idx].cpu().detach().numpy()
        top_class = top_class[sel_boxes_idx].cpu().detach().numpy()
        for j in range(rgb.shape[0]):
            for k in range(boxes.shape[0]):
                pred_info = []
                pred_info.append(batch_idx * self.hparams.batch_size + j)
                pred_info.append(top_class[k])
                pred_info.append(scores[k])
                pred_info.extend(boxes[k].tolist())
                self.preds.append(pred_info)

        return {"a": None}

    def on_validation_epoch_end(self):
        # outs is a list of whatever you returned in `validation_step`
        ap = float(mean_average_precision(self.preds, self.GTs, num_classes=self.num_classes, iou_threshold=0.5))
        self.GTs = []
        self.preds = []
        self.log("val_ap", ap)

    def to_eval(self, device):
        self.model.eval()
        self.model.to(device)

    def detect(self, img, xyz, device, nms_threshold=0.5):
        """Perform inference detection and re-ID feature extraction on images
        - img: torch.tensor of shape (3,H, W)
        - xyz: torch.tensor of shape (3, H, W)
        - device to load the tensors and model
        - nms_threshold: non-maximum suppresion threshold
        """
        height, width = img.shape[1], img.shape[2]

        outputs = self.model(img.unsqueeze(0).to(device), xyz.unsqueeze(0).to(device))
        prob = F.softmax(outputs["pred_logits"][0], dim=1)
        top_p, top_class = prob.topk(1, dim=1)

        prob_tracks = F.softmax(outputs["pred_tracks"][0], dim=1)
        _, track_class = prob_tracks.topk(1, dim=1)
        boxes = outputs["pred_boxes"][0][top_class.squeeze() != self.num_classes]
        track_class = track_class[top_class != self.num_classes]
        features = outputs["features"][0][top_class.squeeze() != self.num_classes]
        scores = top_p[top_class != self.num_classes]
        top_class = top_class[top_class != self.num_classes]

        sel_boxes_idx = torchvision.ops.nms(boxes=box_cxcywh_to_xyxy(boxes), scores=scores, iou_threshold=nms_threshold)

        boxes = box_cxcywh_to_xyxy(boxes[sel_boxes_idx]).cpu().detach().numpy()
        boxes[:, 0] *= width
        boxes[:, 1] *= height
        boxes[:, 2] *= width
        boxes[:, 3] *= height
        scores = scores[sel_boxes_idx].cpu().detach().numpy()
        top_class = top_class[sel_boxes_idx].cpu().detach().numpy()

        track_class = track_class[sel_boxes_idx].cpu().detach().numpy()
        features = features[sel_boxes_idx].cpu().detach()

        return boxes, top_class, scores, track_class, features
