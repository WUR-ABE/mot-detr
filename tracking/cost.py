import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from scipy.spatial.distance import mahalanobis
from torchmetrics.functional import pairwise_cosine_similarity
import torch


def mahalanobis_distance(mean, covariance, measurements):
    distances = []
    for measurement in measurements:
        distances.append(mahalanobis(measurement, mean, np.array(np.matrix(covariance).I)))
    distances = np.square(np.array(distances))
    # distances = distances/distances.max()
    return distances


def mahalanobis_cost_matrix(row_positions, row_covariances, col_positions, gating=True):
    mahalanobis_cost_matrix = np.zeros((len(row_positions), len(col_positions)))
    for i, point in enumerate(row_positions):
        mahalanobis_cost_matrix[i, :] = mahalanobis_distance(point, row_covariances[i], col_positions)
    if gating:
        gating_threshold_pos = 7.8147
        # mahalanobis_cost_matrix = mahalanobis_cost_matrix / mahalanobis_cost_matrix.max()
        mahalanobis_cost_matrix[mahalanobis_cost_matrix > gating_threshold_pos] += 1e5
    return mahalanobis_cost_matrix


def euclidean_cost_matrix(row_positions, col_positions, gating=None):
    euclidean_cost_matrix = distance_matrix(np.array(row_positions), np.array(col_positions))
    if gating:
        idx = np.where(euclidean_cost_matrix > gating)
        euclidean_cost_matrix[idx] += 1e5
        # euclidean_cost_matrix = np.square(euclidean_cost_matrix)
    return euclidean_cost_matrix


def class_cost_distance(row_classes, col_classes):
    class_cost_matrix = np.zeros((len(row_classes), len(col_classes)))
    col_classes = np.array(col_classes)
    for i, row_class in enumerate(row_classes):
        idx = np.where(col_classes != row_class)
        class_cost_matrix[i, idx] += 1e5
    return class_cost_matrix


def cosine_cost_matrix(row_feats, col_feats):
    return (
        1 - pairwise_cosine_similarity(torch.cat(row_feats, dim=0), torch.cat(col_feats, dim=0)).cpu().detach().numpy()
    )


def cosine_cost_matrix_from_list(row_feats, col_feats, gating_max=0.8, gating_min=0.4):
    cost_matrix_max = np.zeros((len(row_feats), len(col_feats)))
    cost_matrix_min = np.zeros((len(row_feats), len(col_feats)))

    for i, track_feats in enumerate(row_feats):
        temp_mat = cosine_cost_matrix(track_feats, col_feats)
        cost_matrix_max[i, :] = temp_mat.max(axis=0)
        cost_matrix_min[i, :] = temp_mat.min(axis=0)

    cost_matrix_min[cost_matrix_min > gating_min] += 1e5
    cost_matrix_min[cost_matrix_max > gating_max] += 1e5

    return cost_matrix_min


def min_cost_matching(
    cost_matrix,
    max_distance,
    track_indices=None,
    detection_indices=None,
):
    rows, cols = linear_sum_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in cols:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in rows:
            unmatched_tracks.append(track_idx)
    for row, col in zip(rows, cols):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] >= max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections
