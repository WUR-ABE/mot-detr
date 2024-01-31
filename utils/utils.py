import torch
import torch.nn.functional as F
import torchvision
from torchvision.ops.boxes import box_area
from collections import Counter


def box_iou(boxes1, boxes2):
    """
    Compute the Intersection over Union (IoU) between two sets of bounding boxes.
    The format of the bounding boxes should be in (x1, y1, x2, y2) format.

    Args:
        boxes1 (torch.Tensor): A tensor of shape (N, 4) in (x1, y1, x2, y2) format.
        boxes2 (torch.Tensor): A tensor of shape (M, 4) in (x1, y1, x2, y2) format.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple of (iou, union) where:
            iou (torch.Tensor): A tensor of shape (N, M) containing the pairwise IoU values
            between the boxes in boxes1 and boxes2.
            union (torch.Tensor): A tensor of shape (N, M) containing the pairwise union
            areas between the boxes in boxes1 and boxes2.
    """
    # Calculate boxes area
    area1 = box_area(boxes1)  # [N,]
    area2 = box_area(boxes2)  # [M,]

    # Compute the coordinates of the intersection of each pair of bounding boxes
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    # Need clamp(min=0) in case they do not intersect, then we want intersection to be 0
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    # Since the size of the variables is different, pytorch broadcast them
    # area1[:, None] converts size from [N,] to [N,1] to help broadcasting
    union = area1[:, None] + area2 - inter  # [N,M]

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Computes the generalized box intersection over union (IoU) between two sets of bounding boxes.
    The IoU is defined as the area of overlap between the two bounding boxes divided by the area of union.

    Args:
        boxes1: A tensor containing the coordinates of the bounding boxes for the first set.
            Shape: [batch_size, num_boxes, 4]
        boxes2: A tensor containing the coordinates of the bounding boxes for the second set.
            Shape: [batch_size, num_boxes, 4]

    Returns:
        A tensor containing the generalized IoU between `boxes1` and `boxes2`.
            Shape: [batch_size, num_boxes1, num_boxes2]
    """
    # Check for degenerate boxes that give Inf/NaN results
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    # Calculate the IoU and union of each pair of bounding boxes
    # TODO: put your code here (~1 line)
    iou, union = box_iou(boxes1, boxes2)

    # Compute the coordinates of the intersection of each pair of bounding boxes
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)  # [N,M,2]

    # Compute the area of the bounding box that encloses both input boxes
    C = wh[:, :, 0] * wh[:, :, 1]

    # TODO: put your code here (~1 line)
    return iou - (C - union) / C


def box_cxcywh_to_xyxy(x):
    """
    Convert bounding boxes from (center x, center y, width, height) format to
    (x1, y1, x2, y2) format.

    Args:
        x (torch.Tensor): A tensor of shape (N, 4) in (center x, center y,
            width, height) format.

    Returns:
        torch.Tensor: A tensor of shape (N, 4) in (x1, y1, x2, y2) format.
    """
    x_c, y_c, w, h = x.unbind(-1)
    x1 = x_c - 0.5 * w
    y1 = y_c - 0.5 * h
    x2 = x_c + 0.5 * w
    y2 = y_c + 0.5 * h
    b = torch.stack([x1, y1, x2, y2], dim=-1)
    return b


def box_xyxy_to_cxcywh(x):
    """
    Convert bounding boxes from (x1, y1, x2, y2) format to (center_x, center_y, width, height)
    format.

    Args:
        x (torch.Tensor): A tensor of shape (N, 4) in (x1, y1, x2, y2) format.

    Returns:
        torch.Tensor: A tensor of shape (N, 4) in (center_x, center_y, width, height) format.
    """
    x0, y0, x1, y1 = x.unbind(-1)
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    width = x1 - x0
    height = y1 - y0
    b = torch.stack([center_x, center_y, width, height], dim=-1)
    return b


def box_xywh_to_xyxy(x):
    """
    Convert bounding box from (x, y, w, h) format to (x1, y1, x2, y2) format.

    Args:
        x (torch.Tensor): A tensor of shape (N, 4) in (x, y, w, h) format.

    Returns:
        torch.Tensor: A tensor of shape (N, 4) in (x1, y1, x2, y2) format.
    """
    x_min, y_min, w, h = x.unbind(-1)
    x_max = x_min + w
    y_max = y_min + h
    b = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    return b


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=20):
    """
    Calculates mean average precision
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    if len(pred_boxes) == 0 or len(true_boxes) == 0:
        return 0.0

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou, _ = box_iou(
                    box_cxcywh_to_xyxy(torch.tensor(detection[3:]).unsqueeze(0)),
                    box_cxcywh_to_xyxy(torch.tensor(gt[3:]).unsqueeze(0)),
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


class APCalculator:
    """A class for calculating average precision (AP).

    This class is built to be used in a training loop, allowing the ground truth (GTs)
    to be initialized once in the __init__ constructor, and then reused many times by calling
    the `calculate_map()` method.

    Attributes:
        iou_threshold (float): The intersection over union (IoU) threshold used for the AP calculation. Defaults to 0.5.
        data_iter (torch.data.Dataloader): A PyTorch dataloader that provides images and targets (e.g., bounding boxes)
            for the dataset.
        n_classes (int): The number of object classes.
        GTs (List[List[Union[int, float]]]): A list of ground truth_target values for each bounding box in the dataset.
        preds (List[List[Union[int, float]]]): A list of predicted_target values for each bounding box in the dataset.

    Args:
        data_iter (torch.data.Dataloader): A PyTorch dataloader that provides images and targets (e.g., bounding boxes)
            for the dataset.
        n_classes (int): The number of object classes.
        iou_threshold (float, optional): The intersection over union (IoU) threshold used for the AP calculation.
            Defaults to 0.5.
    """

    def __init__(self, data_iter, n_classes, iou_threshold=0.5):
        """Initializes the APCalculator object with the specified data iterator, number of classes,
        and IoU threshold."""
        self.iou_threshold = iou_threshold
        self.data_iter = data_iter
        self.n_classes = n_classes
        self.GTs = []

        # Get ground truth target values for each bounding box in the dataset
        for i, (images, targets) in enumerate(self.data_iter):
            new_targets = []
            for idx in range(targets["labels"].shape[0]):
                labels = targets["labels"][idx]
                boxes = targets["boxes"][idx]
                new_targets.append(
                    {
                        "labels": labels[labels != -1].cpu().detach().numpy(),
                        "boxes": boxes[labels != -1].cpu().detach().numpy(),
                    }
                )

            for j in range(images.shape[0]):
                for k in range(new_targets[j]["labels"].shape[0]):
                    label_info = []
                    label_info.append(i * self.data_iter.batch_size + j)  # image index
                    label_info.append(new_targets[j]["labels"][k])  # class label
                    label_info.append(1)  # class label
                    label_info.extend(new_targets[j]["boxes"][k].tolist())  # bounding box coordinates
                    self.GTs.append(label_info)

    def calculate_map(self, net, nms_threshold=0.1):
        """Calculates the mean average precision (mAP) for the given object detection network.

        Args:
            net (torch.nn.Module): The object detection network.
            nms_threshold (float, optional): The non-maximum suppression (NMS) threshold. Defaults to 0.1.

        Returns:
            float: The mean average precision (mAP) for the given network and ground truth targets.
        """
        net.eval()
        preds = []

        for i, (images, targets) in enumerate(self.data_iter):
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            images = images.to(device)
            outputs = net(images)
            outputs["pred_logits"] = outputs["pred_logits"].cpu()
            outputs["pred_boxes"] = outputs["pred_boxes"].cpu()

            prob = F.softmax(outputs["pred_logits"][0], dim=1)
            top_p, top_class = prob.topk(1, dim=1)
            # top_class = torch.where(top_p > 0.7, top_class, self.n_classes)

            boxes = outputs["pred_boxes"][0][top_class.squeeze() != self.n_classes]
            scores = top_p[top_class != self.n_classes]
            top_class = top_class[top_class != self.n_classes]

            sel_boxes_idx = torchvision.ops.nms(
                boxes=box_cxcywh_to_xyxy(boxes), scores=scores, iou_threshold=nms_threshold
            )

            boxes = boxes[sel_boxes_idx].cpu().detach().numpy()
            scores = scores[sel_boxes_idx].cpu().detach().numpy()
            top_class = top_class[sel_boxes_idx].cpu().detach().numpy()
            for j in range(images.shape[0]):
                for k in range(boxes.shape[0]):
                    pred_info = []
                    pred_info.append(i * self.data_iter.batch_size + j)
                    pred_info.append(top_class[k])
                    pred_info.append(scores[k])
                    pred_info.extend(boxes[k].tolist())
                    preds.append(pred_info)

        self.preds = preds
        return float(
            mean_average_precision(preds, self.GTs, num_classes=self.n_classes, iou_threshold=self.iou_threshold)
        )


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for
    the specified values of k.
    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.
    Returns:
        list: accuracy at top-k.
    Examples::
        >>> from torchreid import metrics
        >>> metrics.accuracy(output, target)
    """
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res
