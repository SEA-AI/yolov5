import numpy as np
from typing import Tuple


def cxcywh_to_xyxy(bboxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from (center_x, center_y, width, height) to 
    (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format.

    Parameters
    ----------
    bboxes : np.ndarray (N, 4)
        bounding boxes in (center_x, center_y, width, height) format

    Returns
    -------
    np.ndarray (N, 4)
        bounding boxes in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format
    """
    center_x, center_y, width, height = bboxes.T
    top_left_x = center_x - width / 2
    top_left_y = center_y - height / 2
    bottom_right_x = center_x + width / 2
    bottom_right_y = center_y + height / 2

    return np.array([top_left_x, top_left_y, bottom_right_x, bottom_right_y]).T


def xyxy_to_xywh(bboxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format
    to (center_x, center_y, width, height) format.

    Parameters
    ----------
    bboxes : np.ndarray (N, 4)
        bounding boxes in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format

    Returns
    -------
    np.ndarray (N, 4)
        bounding boxes in (top_left_x, top_left_y, width, height) format
    """
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = bboxes.T
    width = bottom_right_x - top_left_x
    height = bottom_right_y - top_left_y
    return np.array([top_left_x, top_left_y, width, height]).T


def xyxy_to_xyxyn(bboxes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert bounding boxes in form of (x, y, x, y) from absolute pixels to relative pixels
    in range [0, 1] with respect to certain shape.

    Parameters
    ----------
    bboxes : np.ndarray (N, 4)
        bounding boxes in (x, y, x, y) format
    shape : Tuple[int, int] 
        shape in (height, width) to which should be normalized

    Returns
    -------
    np.ndarray (N, 4)
        bounding boxes in (x, y, x, y) format in range [0, 1]
    """
    h, w = shape
    bboxes[:, [0, 2]] /= w
    bboxes[:, [1, 3]] /= h
    return bboxes


def xyxyn_to_xyxy(bboxes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert bounding boxes in form of (x, y, x, y) from relative pixels in range [0, 1]
    to absolute pixels with respect to certain shape.

    Parameters
    ----------
    bboxes : np.ndarray (N, 4)
        bounding boxes in (x, y, x, y) format
    shape : Tuple[int, int] 
        shape in (height, width) to which should be normalized

    Returns
    -------
    np.ndarray (N, 4)
        bounding boxes in (x, y, x, y) format in absolute pixels
    """
    h, w = shape
    bboxes[:, [0, 2]] *= w
    bboxes[:, [1, 3]] *= h
    return bboxes


def scale_boxes(bboxes, from_shape, to_shape):
    """
    Rescale bounding boxes from from_shape to to_shape

    Parameters
    ----------
    bboxes : np.ndarray (N, 4)
        bounding boxes in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format
    from_shape : tuple (2,)
        original shape of the image (height, width)
    to_shape : tuple (2,)
        target shape of the image (height, width)

    Returns
    -------
    np.ndarray (N, 4)
        bounding boxes in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format
    """

    # gain  = old / new
    gain = min(from_shape[0] / to_shape[0], from_shape[1] / to_shape[1])
    # wh padding
    pad = ((from_shape[1] - to_shape[1] * gain) / 2,  # x padding
           (from_shape[0] - to_shape[0] * gain) / 2)  # y padding

    bboxes[..., [0, 2]] -= pad[0]  # x padding
    bboxes[..., [1, 3]] -= pad[1]  # y padding
    bboxes[..., :4] /= gain
    clip_boxes(bboxes, to_shape)
    return bboxes


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def intersection_box(target_box: np.ndarray, boxes: np.ndarray) -> list:
    """
    Calculate intersection of target boxes with all others in boxes

    Parameters
    ----------
    target_box : np.ndarray (4,)
        bounding box in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format
    boxes : np.ndarray (N, 4)
        bounding boxes in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format

    Returns
    -------
    np.ndarray (N, 4)
        intersection bounding boxes in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format
    """
    x1, y1, x2, y2 = target_box
    a1, b1, a2, b2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    xx1 = np.maximum(x1, a1)
    yy1 = np.maximum(y1, b1)
    xx2 = np.minimum(x2, a2)
    yy2 = np.minimum(y2, b2)

    return np.array([xx1, yy1, xx2, yy2])


def intersection_area(target_box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    calculate intersection of target boxes with all others in boxes
    Returns:
        (np.ndarray) areas of intersection bounding boxes 
    """
    xx1, yy1, xx2, yy2 = intersection_box(target_box, boxes)
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    return w*h


def iou(target_box: np.ndarray,
        boxes: np.ndarray,
        target_area: np.ndarray,
        areas: np.ndarray) -> np.ndarray:
    """
    Calculate intersection over union of target box with all others

    Parameters
    ----------
    target_box : np.ndarray (4,)
        bounding box in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format
    boxes : np.ndarray (N, 4)
        bounding boxes in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format
    target_area : np.ndarray (1,)
        area of target box
    areas : np.ndarray (N,)
        areas of all boxes
    """
    inter = intersection_area(target_box, boxes)
    union = target_area + areas - inter

    # avoid division w/ 0
    union[union == 0] += 1e-7

    return inter / union


def nms(bboxes: np.ndarray, scores: np.ndarray, iou_thresh: float, classes: np.ndarray = None) -> np.ndarray:
    """
    Non-Maximum-Supression (NMS) algorithm to remove overlapping bounding boxes.

    Parameters
    ----------
    bboxes : np.ndarray (N, 4)
        bounding boxes in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format
    scores : np.ndarray (N,)
        confidence scores for each bounding box
    iou_thresh : float
        threshold for intersection over union
    classes : np.ndarray (N,)
        class labels for each bounding box
        if None, all boxes are treated as the same class (agnostic NMS)
    """
    bboxes = bboxes.astype(np.float32)
    x1, y1 = bboxes[..., 0], bboxes[..., 1]  # top left
    x2, y2 = bboxes[..., 2], bboxes[..., 3]  # bottom right

    areas = (x2 - x1) * (y2 - y1)  # calculate area per bbox
    order = np.argsort(scores)[::-1]  # sort by descending confidence

    keep = np.bool_(np.zeros_like(scores))  # # which boxes (idx) to keep

    while order.size > 1:
        # take current highest confidence box
        i = order[0]
        keep[i] = True

        # calculate intersection over union with all other boxes
        ovr = iou(bboxes[i], bboxes[order[1:]], areas[i], areas[order[1:]])

        # find boxes to keep since they are not overlapping enough
        idxs = np.where(ovr <= iou_thresh)[0]

        if classes:
            # only keep boxes with different classes
            idxs = np.union1d(idxs, np.where(classes[order[1:]] != classes[i]))

        # update order by removing suppressed boxes
        order = order[idxs + 1]  # +1 because we removed the first element

    return keep
