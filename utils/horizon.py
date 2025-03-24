from typing import Tuple

import cv2
import numpy as np
import torch


def postprocess_x_pitch_theta(x_pitch, x_theta):
    """
    TODO: refactor postprocessing
    """
    # get maximum pitch and theta and normalise
    x_pitch, x_theta = x_pitch.softmax(1), x_theta.softmax(1)
    conf_pitch, pitch = x_pitch.max(1, keepdim=True)
    pitch = pitch / x_pitch.shape[-1]
    conf_theta, theta = x_theta.max(1, keepdim=True)
    theta = theta / x_theta.shape[-1]
    return torch.cat((pitch, conf_pitch), 1), torch.cat((theta, conf_theta), 1)


def points_to_pitch_theta(x1: float, y1: float, x2: float, y2: float):
    """
    Parametrize line that is spanned by the two points (x1, y1) and (x2, y2) through pitch and theta.

       - pitch ... y-value of line where x=0.5 (pitch=0 is at the bottom, pitch=1 is at the top)
       - theta ... angle between line and y=0.5 (aka horizontal line)

       Assumptions:
       Coordinates are in range [0, 1].
       Origin is top left corner.

    Args:
        x1 (float): x-coordinate of point 1
        y1 (float): y-coordinate of point 1
        x2 (float): x-coordinate of point 2
        y2 (float): y-coordinate of point 1

    Returns:
        pitch (float): in [0,1] (pitch=0.25 is bottom, pitch=0.75 is top)
        theta (float): in [0,1] (theta=0 is -pi/2, theta=1 is pi/2)
    """
    assert x1 != x2, "Line is not allowed to be perfectly vertical or pitch would be infinite"

    if x1 > x2:  # make sure x1 < x2
        x1, y1, x2, y2 = x2, y2, x1, y1

    # invert y-axis since origin is top left corner
    y1 = 1 - y1
    y2 = 1 - y2

    m, b = points_to_slope_intercept(x1, y1, x2, y2)
    pitch = m * 0.5 + b
    theta = np.arctan(m)  # rad [-pi/2, pi/2]

    # "encode" to [0, 1]
    pitch = 0.7 * pitch + 0.15  # [0.15, 0.85] is inside the image
    theta = (theta + 0.5 * np.pi) / np.pi  # [0, 1]

    return pitch, theta


def pitch_theta_to_slope_intercept(pitch: float, theta: float):
    """
    Convert pitch and theta to slope-intercept form of line: m, b.

    Args:
        pitch (float): in [0,1] (pitch=0.15 is bottom, pitch=0.85 is top)
        theta (float): in [0,1] (theta=0 is -pi/2, theta=1 is pi/2)

    Returns:
        m, b
    """

    # "decode" from [0, 1]
    pitch = (pitch - 0.15) / 0.7  # [0, 1]
    theta = theta * np.pi - 0.5 * np.pi  # rad [-pi/2, pi/2]

    m = np.tan(theta)
    b = pitch - m * 0.5
    return m, b


def pitch_theta_to_points(pitch: float, theta: float, input_hw, orig_hw):
    """
    Convert pitch and theta to two points.

    Args:
        pitch (float): in [0,1] (pitch=0.25 is bottom, pitch=0.75 is top)
        theta (float): in [0,1] (theta=0 is -pi/2, theta=1 is pi/2)
        input_hw (tuple): input height and width
        orig_hw (tuple): original height and width

    Returns:
        (x1, y1), (x2, y2)
    """
    m, b = pitch_theta_to_slope_intercept(pitch, theta)
    points = slope_intercept_to_points(m, b, input_hw[1], input_hw[0])
    points = scale_line_edges(np.array([points]).flatten(), input_hw, orig_hw, upscale=False)
    (x1, y1), (x2, y2) = points.reshape(-1, 2)
    y1, y2 = orig_hw[0] - y1, orig_hw[0] - y2
    return (x1, y1), (x2, y2)


def scale_line_edges(
    line_edges: np.ndarray,
    from_shape: Tuple[int, int],
    to_shape: Tuple[int, int],
    upscale: bool = True,
) -> np.ndarray:
    """
    Rescale bounding boxes from from_shape to to_shape.

    Parameters
    ----------
    bboxes : np.ndarray (N, 4)
        bounding boxes as (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    from_shape : tuple (2,)
        original shape of the image (height, width)
    to_shape : tuple (2,)
        target shape of the image (height, width)
    upscale : bool
        whether to upscale the bounding boxes

    Returns
    -------
    np.ndarray (N, 4)
        bounding boxes as (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    """
    # gain  = old / new
    gain = min(from_shape[0] / to_shape[0], from_shape[1] / to_shape[1])
    gain = min(gain, 1) if not upscale else gain
    # wh padding
    pad = (
        (from_shape[1] - to_shape[1] * gain) / 2,  # x padding
        (from_shape[0] - to_shape[0] * gain) / 2,
    )  # y padding

    line_edges[..., [0, 2]] -= pad[0]  # x padding
    line_edges[..., [1, 3]] -= pad[1]  # y padding
    line_edges[..., :4] /= gain
    return line_edges


def slope_intercept_to_points(m: float, b: float, w: int = 1, h: int = 1):
    # to m, b in normalised space
    if m == np.inf:
        x_1, y_1 = b, 0
        x_2, y_2 = b, 1
    else:
        x_1, y_1 = 0, b
        x_2, y_2 = 1, m + b
    # scale to image space
    return (x_1 * w, y_1 * h), (x_2 * w, y_2 * h)


def points_to_hough(x_1, y_1, x_2, y_2):
    """Convert two points to hough form of line: rho, theta."""
    if x_1 == x_2:  # vertical line
        return x_1, 0
    m = (y_2 - y_1) / (x_2 - x_1)
    a, b, c = -m, 1, y_1 - m * x_1
    rho = (y_1 - m * x_1) / np.sqrt(a**2 + b**2)
    theta = np.arctan2(b, a)
    return rho, theta


def points_to_slope_intercept(x_1, y_1, x_2, y_2):
    """Convert two points to slope-intercept form of line: m, b."""
    if x_1 == x_2:  # vertical line
        return np.inf, x_1
    m = (y_2 - y_1) / (x_2 - x_1)
    b = y_1 - m * x_1
    return m, b


def hough_to_points(rho, theta, h=1, w=1):
    """
    Convert hough form of line to two points. Points are located on the image border. Points are normalised unless h and
    w are provided.

    hough form --> slope-intercept form --> points.
    """
    m, b = hough_to_slope_intercept(rho, theta, h, w)
    (x_1, y_1), (x_2, y_2) = slope_intercept_to_points(m, b, w, h)
    return (x_1, y_1), (x_2, y_2)


def hough_to_slope_intercept(rho, theta, h=1, w=1):
    """Convert hough form of line to slope-intercept form."""
    if theta == 0:
        return np.inf, rho * w
    if theta == np.pi / 2:
        return 0, rho * h
    a, b, c = np.cos(theta), np.sin(theta), rho
    m = -a / b
    b = -c / b
    # take into account image dims
    m *= h / w
    b = -b * h
    return m, b


def draw_horizon(image, keypoints=None, pitch_theta=None, hough=None, color=(0, 255, 0), diameter=2):
    """Visualize horizon line on image."""
    assert np.any(
        [arg is not None for arg in [keypoints, pitch_theta, hough]]
    ), "Provide at least one of keypoints, theta_pitch, hough"

    image = image.copy()

    if keypoints is not None:
        x1, y1, x2, y2 = np.array(keypoints).flatten()
        for x, y in keypoints:
            cv2.circle(image, (int(x), int(y)), diameter * 5, (0, 255, 0), -1)

    if pitch_theta is not None:
        (x1, y1), (x2, y2) = pitch_theta_to_points(*pitch_theta, image.shape, image.shape)

    if hough is not None:
        (x1, y1), (x2, y2) = hough_to_points(*hough, image.shape[1], image.shape[0])

    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, diameter)
    return image


def draw_bboxes(image, dets, color=(0, 255, 0), thickness=2):
    """Visualize bounding boxes on image."""
    image = image.copy()
    for det in dets:
        x1, y1, x2, y2 = det[:4]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return image
