"""
AHOYv5 model for object and horizon detection.

TODO: add metadata to model file:
    - offset_buffer (how offset is encoded in model)
    - classes_names
"""

import logging
import os
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
import yaml

from .misc import Profile
from .postprocessing import postprocess_ahoy, xyxy_to_xyxyn

# import pycuda.autoinit  # cuda context initialized manually in __init__
from .preprocessing import preprocess_yolo, resize_and_center_images_in_batch

logger = logging.getLogger(__name__)


def load_labels(label_path):
    """Load label mapping from file."""
    with open(label_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class AHOYv5:
    """A Horizon and Object detection based on YOLOv5."""

    def __init__(self, model_path: str, device: int = 0) -> None:
        """
        Initialize the AHOYv5 model.

        Parameters
        ----------
        model_path : str
            Path to the model file containing the weights and architecture.
        device : int, optional
            CUDA device to use.
        """
        if Path(model_path).suffix == ".engine":
            from .infertrt import InferTRT

            self.model = InferTRT(model_path, device)
        else:
            raise ValueError("Only TensorRT engines are supported.")

        cls_map_fpath = model_path.replace(".engine", ".yaml")
        if os.path.exists(cls_map_fpath):
            self.cls_map = load_labels(cls_map_fpath)
            logger.info(f"{self.cls_map}")
        else:
            self.cls_map = None

        self.profiles = {
            "preprocess": Profile(),
            "inference": Profile(),
            "postprocess": Profile(),
        }

    def __call__(
        self,
        ims: np.ndarray,
        conf_thresh: float = 0.147,
        iou_thresh: float = 0.1,
        do_curve_fit: bool = True,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Run the model pipeline on the given input image(s).

        Parameters
        ----------
        ims : np.ndarray
            The input image(s).
        conf : float, optional
            Confidence threshold for p(class). Predictions with score < conf_thresh
            are ignored.
        iou_thresh : float, optional
            Minimum IOU to be counted as a duplicate detection.
        do_curve_fit : bool, optional
            If True, uses curve fitting to get the offset and theta values.
            Applicable only for offset-theta model.
        verbose : bool, optional
            If True, prints the time spent in each step of the pipeline.

        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray
            - Predicted bounding boxes.
            - Confidence scores.
            - Class labels (ints)
        """

        orig_shape = ims.shape[-3:-1]  # (h, w)

        with self.profiles["preprocess"]:
            ims = self.preprocess(ims)

        with self.profiles["inference"]:
            output = self.model.forward(ims)

        with self.profiles["postprocess"]:
            preds = self.postprocess(
                output,
                orig_shape,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                do_curve_fit=do_curve_fit,
            )

        if verbose:
            for name, profile in self.profiles.items():
                print(f"{profile.dt * 1E3:>5.1f} ms - {name}")

        return preds

    def preprocess(self, ims: np.array) -> np.ndarray:
        """
        Transform the input image so that the model can infer from it.

        Parameters
        ----------
        ims : np.ndarray
            The input image(s)

        Returns
        -------
        np.ndarray
            Preprocessed image.
        """
        return self.preprocess_fast(ims)

    def preprocess_yolo(self, ims: np.array) -> np.ndarray:
        """
        Transform the input image so that the model can infer from it. Letterbox, 0-1 normalization, and NHWC to NCHW
        conversion.

        Parameters
        ----------
        ims : np.ndarray
            The input image(s)

        Returns
        -------
        np.ndarray
            Preprocessed image.
        """
        return preprocess_yolo(ims, self.model.input_hw, self.model.fp16)

    def preprocess_fast(self, ims: np.array) -> np.ndarray:
        """
        Transform the input image so that the model can infer from it. This runs under the assumption that input images
        have a constant size throughout the pipeline.

        Parameters
        ----------
        ims : np.ndarray
            The input image(s)

        Returns
        -------
        np.ndarray
            Preprocessed image.
        """
        return resize_and_center_images_in_batch(ims, self.model.alloc_arrays)

    def postprocess(
        self,
        output: tuple,
        orig_hw: tuple,
        conf_thresh: float,
        iou_thresh: float,
        offset_buffer: float = 0.15,
        do_curve_fit: bool = True,
    ) -> Sequence[np.ndarray]:
        """
        Transform raw model output to application output.

        Parameters
        ----------
        output : tuple
            Raw model output.
        orig_hw : tuple
            Original image shape (height, width)
        conf_thresh : float
            Confidence threshold for p(bbox)
            Predictions with score < conf are not considered as output.
        iou_thresh : float
            Minimum IOU to be counted as a duplicate detection.
        offset_buffer : float, optional
            Buffer for offset-theta model (only used if model has 3 outputs).

        Returns
        -------
        Sequence[np.ndarray]
            Every element is a np.ndarray of shape (N, 6) where:
            - N is the number of detected bounding boxes,
            - first 4 are the coordinates of the bounding box,
            - 5 is the confidence score of the bounding box,
            - 6 is the class label of the detected bounding box.
        """

        return postprocess_ahoy(
            output,
            self.model.input_hw,
            orig_hw,
            conf_thresh,
            iou_thresh,
            offset_buffer,
            do_curve_fit,
        )

    def predict(self, img, conf_thresh, iou_thresh, do_curve_fit=True):
        """Shortcut for detect() with output_mode="qa"."""
        return self.detect(
            img,
            output_mode="qa",
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            do_curve_fit=do_curve_fit,
        )

    def detect(
        self,
        ims: np.ndarray,
        output_mode: Union[str, None] = "qa",
        conf_thresh: float = 0.147,
        iou_thresh: float = 0.1,
        do_curve_fit: bool = False,
    ):
        """
        Parameters
        ----------------------

        - ims: np.ndarray
            The input image(s).
        - output_mode: str, default "qa"
            If "tf", returns format expected by Tensorflow Object Detection API::

                return {
                    "detection_boxes": bboxes,
                    "detection_scores": scores,
                    "detection_classes": classes,
                    "num_detections": len(bboxes),
                }

            If "qa", returns format expected by the QA system::

                return [[[x1, y1, x2, y2], class_name, class_id, score], ...]

            Otherwise, returns (bboxes, scores, classes)

        - conf_thresh: float, optional
            Confidence threshold for p(class). Predictions with score<conf are ignored.
        - iou_thresh: float, optional
            Minimum IOU to be counted as a duplicate detection.
        - do_curve_fit: bool, optional
            If True, uses curve fitting to get the offset and theta values.
            Applicable only for offset-theta model.
        """
        if isinstance(ims, np.ndarray):
            assert ims.ndim == 4, "Input must have 4 dimensions (b, h, w, c)"
        else:
            raise ValueError("Input must be a numpy array. Got: ", type(ims))

        orig_shape = ims.shape[-3:-1]  # (h, w)
        dets = self(ims, conf_thresh, iou_thresh, do_curve_fit)
        return self.to_mode(dets, output_mode, orig_shape, self.cls_map)

    @staticmethod
    def to_mode(dets, output_mode, orig_shape, cls_map):
        """Convert detections to desired output mode."""
        bboxes = [det[:, :4] for det in dets]
        scores = [det[:, 4] for det in dets]
        classes = [det[:, 5] for det in dets]

        if output_mode == "tf":
            return AHOYv5.to_tf(bboxes, scores, classes, orig_shape)

        if output_mode == "qa":
            return AHOYv5.to_qa(bboxes, scores, classes, orig_shape, cls_map)

        return bboxes, scores, classes

    @staticmethod
    def to_tf(bboxes, scores, classes, orig_shape) -> List[dict]:
        """To Tensorflow Object Detection API format."""
        proposals = []
        for bbs, scs, cls in zip(bboxes, scores, classes):
            bbs = xyxy_to_xyxyn(bbs, orig_shape)
            bbs = bbs[:, [1, 0, 3, 2]]
            proposals.append(
                {
                    "detection_boxes": bbs,
                    "detection_scores": scs,
                    "detection_classes": cls,
                    "num_detections": len(bbs),
                }
            )
        return proposals

    @staticmethod
    def to_qa(bboxes, scores, classes, orig_shape, cls_map) -> List[List[list]]:
        """
        Named 'qa' for some historical reason since the QA department used this format.

        Format is [[[y1, x1, y2, x2], class_name, class_id, score], ...]
        """
        if not cls_map:
            raise ValueError("Class mapping not found. Cannot convert to QA format.")
        proposals = []
        for bbs, scs, cls in zip(bboxes, scores, classes):
            bbs = xyxy_to_xyxyn(bbs, orig_shape)
            proposals.append([[b, cls_map[c], int(c), s] for b, s, c in zip(bbs, scs, cls)])
        return proposals

    def close(self):
        """Run this before exiting the program to free up resources."""
        self.model.close()
