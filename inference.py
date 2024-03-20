import numpy as np
import torch
from fte_utils.pointclouds import PointCloud
import polars as pl


from torchvision.transforms import ToTensor

from tracking.world_model import WorldModel
from tracking.detection import Detection

from models.mot_detr import MOTDETR
from datasets.dataset import channel_to_heatmap


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class App:
    def __init__(self, feat_checkpoint, w, gating_min, gating_max, n_init=0, R_k=0.01):
        self.detector = MOTDETR.load_from_checkpoint(feat_checkpoint)
        self.detector.to_eval(device)

        self.to_tensor = ToTensor()

        # Multi object tracking initialization
        self.world_model = WorldModel(n_init=n_init, w=w, gating_min=gating_min, gating_max=gating_max)
        self.R = np.eye(3, 3) * R_k

        self.frame_number = 1
        self.results_track = []

    def run(self, pcd_path, debug=False, t_noise=0.005):
        pcd = PointCloud.from_pcd(pcd_path)
        pcd.save_binary(f"vis/pcds/{self.frame_number}.pcd")
        rgb, xyz_ori = pcd.get_points_rgb_xyz_array(reshape=True)

        xyz = np.nan_to_num(xyz_ori, nan=0.0)

        # synthetic
        xyz[:, :, 0] = channel_to_heatmap(xyz[:, :, 0], min=-0.33775, max=0.34639)
        xyz[:, :, 1] = channel_to_heatmap(xyz[:, :, 1], min=-0.34417, max=0.35452)
        xyz[:, :, 2] = channel_to_heatmap(xyz[:, :, 2], min=-0.09238, max=2.26342)

        # real tomato
        # xyz[:, :, 0] = channel_to_heatmap(xyz[:, :, 0], min=-0.4, max=0.8)
        # xyz[:, :, 1] = channel_to_heatmap(np.absolute(xyz[:, :, 1]), min=0, max=0.8)
        # xyz[:, :, 2] = channel_to_heatmap(xyz[:, :, 2], min=0.2, max=1.5)

        boxes, top_class, scores, _, features = self.detector.detect(
            self.to_tensor(rgb), self.to_tensor(xyz), device="cuda", nms_threshold=0.4
        )

        det_list = []
        if len(boxes) > 0:
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                p1 = int(cx - w * 0.25)
                p2 = int(cx + w * 0.25)
                p3 = int(cy - h * 0.25)
                p4 = int(cy + h * 0.25)

                xyz_filter = xyz_ori[p3:p4, p1:p2, :].reshape(-1, 3)
                xyz_filter = xyz_filter[~np.isnan(xyz_filter).any(axis=1)]
                z = xyz_filter.mean(axis=0)
                if xyz_filter.shape[0] > 0:
                    det = Detection(
                        z,
                        self.R,
                        top_class[idx],
                        box,
                        scores[idx],
                        features[idx],
                    )
                    det_list.append(det)
        self.world_model.run(det_list)

        true_tracks = [track for track in self.world_model.tracker.tracks if track.is_confirmed()]
        for track in true_tracks:
            if track.time_since_update == 0:
                self.results_track.append(
                    {
                        "frame_number": self.frame_number,
                        "id": track.track_id,
                        "bb_left": track.last_bbox[0],
                        "bb_top": track.last_bbox[1],
                        "bb_width": track.last_bbox[2] - track.last_bbox[0],
                        "bb_height": track.last_bbox[3] - track.last_bbox[1],
                        "conf": track.track_last_conf,
                        "x": -1,
                        "y": -1,
                        "z": -1,
                    }
                )
        self.frame_number += 1

    def get_results_df(self):
        if len(self.results_track) > 0:
            results_track_df = pl.DataFrame(self.results_track)
            results_track_df = results_track_df.with_columns(
                [
                    pl.col("bb_left").round(2),
                    pl.col("bb_top").round(2),
                    pl.col("bb_width").round(2),
                    pl.col("bb_height").round(2),
                    pl.col("conf").round(2),
                ]
            )
        else:
            results_track_df = pl.DataFrame([{}])

        return results_track_df
