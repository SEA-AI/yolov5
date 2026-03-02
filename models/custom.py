from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from ultralytics import YOLO as UltralyticsYOLO
from ultralytics.nn.tasks import BaseModel as UBaseModel

from models.common import Classify, DetectMultiBackend
from models.experimental import attempt_load
from models.yolo import BaseModel, Detect, DetectionModel
from utils.export import get_weights_path, transform_sz
from utils.general import LOGGER, scale_boxes
from utils.plots import feature_visualization
from utils.torch_utils import is_obb_weights, select_device


class OBBModel(UBaseModel):
    """Wrapper around yolo-OBB Model."""

    def __init__(
        self,
        weights: str = "yolov11n-obb.pt",
        device: Union[str, torch.device] = "",  # automatically select device
        fp16: bool = False,
        fuse: bool = False,
    ):
        super().__init__()
        self.device = select_device(str(device))
        self.fp16 = fp16

        if Path(weights).is_file() or weights.endswith(".pt"):
            model = UltralyticsYOLO(model=weights)
            LOGGER.info(f"Loaded weights from {weights}")
        else:
            raise ValueError("model must be a path to a .pt file")

        model.to(self.device)
        if fuse:
            model.fuse()
        if fp16:
            model.half()
        else:
            model.float()
        self.names = model.names
        self.stride = model.model.stride
        self.save = model.model.save
        self.model = model.model.model  # YOLO.OBBModel.Sequential

    def prepare_for_export(self, dynamic: bool = False):
        """Prepare model for export."""
        from ultralytics.nn import modules  # import C2f, Classify, Detect, RTDETRDecoder

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        for m in self.model.modules():
            if isinstance(m, modules.Classify):
                m.export = True
            if isinstance(
                m, (modules.Detect, modules.RTDETRDecoder)
            ):  # includes all Detect subclasses like Segment, Pose, OBB
                m.dynamic = dynamic
                m.export = True
                m.format = "onnx"
                m.max_det = 1000
                m.xyxy = True  # self.args.nms and not coreml
            elif isinstance(m, modules.C2f):
                # EdgeTPU does not support FlexSplitV while split provides cleaner ONNX graph
                m.forward = m.forward_split

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")


class HorizonModel(BaseModel):
    """YOLOv5 backbone + classification heads for pitch and theta."""

    def __init__(
        self,
        weights: str = "yolov5n.pt",
        nc_pitch: int = 500,
        nc_theta: int = 500,
        device: Union[str, torch.device] = "",  # automatically select device
        cutoff: int = None,
        fp16: bool = False,
        fuse: bool = False,  # false for training, true for inference
    ):
        """
        Horizon detection model.

        Args:
            model (DetectionModel): YOLOv5 model
            nc_pitch (int, optional): number of classes for pitch classification. Defaults to 500.
            nc_theta (int, optional): number of classes for theta classification. Defaults to 500.
            device (str, optional): device to run model on. Defaults to ''.
            cutoff (int, optional): cutoff layer for classification heads.
                If not specified, the SPPF layer is used as cutoff. Defaults to None.
        """
        super().__init__()

        assert weights is not None, "weights must be specified"
        assert isinstance(nc_pitch, int), "nc_pitch must be an integer"
        assert isinstance(nc_theta, int), "nc_theta must be an integer"
        assert isinstance(fp16, bool), "fp16 must be a boolean"

        self.nc_pitch = nc_pitch
        self.nc_theta = nc_theta
        self.device = select_device(str(device))
        self.fp16 = fp16

        if Path(weights).is_file() or weights.endswith(".pt"):
            model = attempt_load(weights, device="cpu", fuse=fuse)
            stride = model.stride
            LOGGER.info(f"Loaded weights from {weights}")
        else:
            raise ValueError("model must be a path to a .pt file")

        if isinstance(model, DetectionModel):
            LOGGER.warning("WARNING ⚠️ converting YOLOv5 DetectionModel to HorizonModel")
            self.cutoff = _find_cutoff(model) if cutoff is None else cutoff
            self._from_detection_model(model, self.cutoff)  # inplace modification

        self.model = model.model
        self.model.to(self.device)
        if fp16:
            self.model.half()
        else:
            self.model.float()
        self.stride = stride
        self.save = model.save

    def _from_detection_model(self, model: DetectionModel, cutoff: int):
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend

        c_pitch, c_theta = _get_classification_heads(model, cutoff, self.nc_pitch, self.nc_theta)
        model.save = set(list(model.save + [cutoff]))  # add cutoff to save

        # remove layers after cutoff
        model.model = model.model[: self.cutoff + 1]

        # add classification heads to model
        model.model.add_module(c_pitch.i, c_pitch)
        model.model.add_module(c_theta.i, c_theta)

    def _forward_once(self, x, profile=False, visualize=False):
        x_pitch, x_theta = None, None
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if m.type == "models.common.Classify" and "pitch" in m.i:
                x_pitch = m(x)
            elif m.type == "models.common.Classify" and "theta" in m.i:
                x_theta = m(x)
            else:
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
                if visualize:
                    feature_visualization(x, m.type, m.i, save_dir=visualize)

        return (x_pitch, x_theta)

    def to_discrete(self, pitch: torch.Tensor, theta: torch.Tensor):
        """
        Take values from [0, 1] and convert to discrete values.

        Values are rounded and clamped to [0, nc_pitch - 1] and [0, nc_theta - 1].
        """
        pitch_i = (pitch * self.nc_pitch).round().clamp(0, self.nc_pitch - 1).long()
        theta_i = (theta * self.nc_theta).round().clamp(0, self.nc_theta - 1).long()
        return pitch_i, theta_i

    def to_continuous(self, pitch_i: torch.Tensor, theta_i: torch.Tensor):
        """
        Take discrete values and convert to continuous values.

        NOTE: Exact values cannot be guaranteed because of rounding.
        """
        pitch = pitch_i.float() / (self.nc_pitch)
        theta = theta_i.float() / (self.nc_theta)
        return pitch, theta

    @staticmethod
    def postprocess(x_pitch: torch.Tensor, x_theta: torch.Tensor):
        """
        Postprocess classification heads.

        Args:
            x_pitch (torch.Tensor): pitch classification head
            x_theta (torch.Tensor): theta classification head

        Returns:
            tuple: (score_pitch, val_pitch), (score_theta, val_theta)
        """
        x_pitch, x_theta = x_pitch.softmax(-1), x_theta.softmax(-1)
        score_pitch, y_pitch = x_pitch.max(-1)
        score_theta, y_theta = x_theta.max(-1)

        # normalise pitch and theta
        y_pitch = y_pitch / x_pitch.size(-1)
        y_theta = y_theta / x_theta.size(-1)

        return (y_pitch, score_pitch), (y_theta, score_theta)

    @staticmethod
    def decode_pitch(offset: float, offset_buffer: float = 0.15):
        """
        Decode offset logits to their original values.

        Parameters
        ----------
            offset (float): in [0,1] (horizon line center at offset=0.5)
            offset_buffer (float): bottom of the image and (1 - offset_buffer) is the
                top of the image. Depends on how the model was trained.

        Returns
        -------
            offset in normalised form [0 - offset_buffer, 1 + offset_buffer]
            where 0 is the bottom of the image and 1 is the top of the image.
        """
        adjusted_offset = offset - offset_buffer
        adjusted_range = 1 - 2 * offset_buffer
        return adjusted_offset / adjusted_range

    @staticmethod
    def decode_theta(theta: float):
        """
        Decode theta logits to their original values.

        Parameters
        ----------
            theta (float): in [0,1] (theta=0 is -pi/2, theta=1 is pi/2)

        Returns
        -------
            theta in radians [-pi/2, pi/2]
        """
        return theta * np.pi - 0.5 * np.pi

    @staticmethod
    def postprocess_curve_fit(x_pitch: torch.Tensor, x_theta: torch.Tensor):
        """
        Postprocess classification heads using Gaussian curve fitting.

        NOTE: Experimental!, supports only batch size of 1.

        Args:
            x_pitch (torch.Tensor): pitch classification head
            x_theta (torch.Tensor): theta classification head

        Returns:
            tuple: (score_pitch, val_pitch), (score_theta, val_theta)
        """
        from scipy.optimize import curve_fit

        # convert to numpy
        x_pitch = x_pitch.squeeze().softmax(-1).cpu().numpy()
        x_theta = x_theta.squeeze().softmax(-1).cpu().numpy()

        # curve to fit
        def gaussian(x, amplitude, mu, sigma):
            return amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

        # Initial guess for the parameters (amplitude, mean, standard deviation)
        initial_pitch_guess = [
            x_pitch.max(),
            x_pitch.argmax() / x_pitch.shape[-1],
            0.001,
        ]
        initial_theta_guess = [
            x_theta.max(),
            x_theta.argmax() / x_theta.shape[-1],
            0.001,
        ]

        x = np.linspace(0, 1, len(x_pitch), endpoint=False)
        fitted_pitch = curve_fit(gaussian, x, x_pitch, p0=initial_pitch_guess)
        fitted_theta = curve_fit(gaussian, x, x_theta, p0=initial_theta_guess)

        amp_pitch, mu_pitch, sigma_pitch = fitted_pitch[0]
        amp_theta, mu_theta, sigma_theta = fitted_theta[0]

        return (mu_pitch, amp_pitch, sigma_pitch), (mu_theta, amp_theta, sigma_theta)

    def prepare_for_export(self, dynamic: bool = False):
        """Prepare model for export."""
        self.model.eval()

        # Update model
        for _, m in self.model.named_modules():
            if isinstance(m, Detect):
                m.inplace = False
                m.dynamic = dynamic
                m.export = True


class ObjectsModel(BaseModel):
    """Wrapper around YOLOv5 DetectionModel."""

    def __init__(
        self,
        weights: str = "yolov5n.pt",
        device: Union[str, torch.device] = "",  # automatically select device
        fp16: bool = False,
        fuse: bool = False,
    ):
        super().__init__()
        self.device = select_device(str(device))
        self.fp16 = fp16

        if Path(weights).is_file() or weights.endswith(".pt"):
            model = attempt_load(weights, device="cpu", fuse=fuse)
            stride = model.stride
            names = model.module.names if hasattr(model, "module") else model.names  # get class names
            LOGGER.info(f"Loaded weights from {weights}")
        else:
            raise ValueError("model must be a path to a .pt file")

        self.model = model.model
        self.model.to(self.device)
        if fp16:
            self.model.half()
        else:
            self.model.float()
        self.stride = stride
        self.names = names
        self.save = model.save

    def prepare_for_export(self, dynamic: bool = False):
        """Prepare model for export."""
        self.model.eval()

        # Update model
        for _, m in self.model.named_modules():
            if isinstance(m, Detect):
                m.inplace = False
                m.dynamic = dynamic
                m.export = True


class YOLO(nn.Module):
    """YOLO model: one or more detection weights, same input, single output.

    - One weight: single detection model with preprocessing/postprocessing hooks.
    - Two or more weights: ensemble (same input to all, merged output with shared class space).
    """

    N_FIXED_COLS = 5  # x1, y1, x2, y2, confidence

    def __init__(
        self,
        weights: Union[str, Sequence[str]],
        device: Union[str, torch.device] = "",
        fp16: bool = False,
        fuse: bool = True,
        imgsz: int | Tuple[int, int] = 640,
        infsz: int | Tuple[int, int] | None = None,
    ):
        """Initialize YOLO. Pass one path for single model, N paths for ensemble (merged output)."""
        super().__init__()

        weights_list = [weights] if isinstance(weights, str) else list(weights)
        if not weights_list:
            raise ValueError("weights must be at least one path")
        self.obj_det_weights = (
            weights_list[0] if len(weights_list) == 1 else weights_list
        )  # saved in onnx model metadata
        self._det_models = nn.ModuleList(
            [ObjectsModel(get_weights_path(w), device=device, fp16=fp16, fuse=fuse) for w in weights_list]
        )

        # First model drives device, stride, and preprocessing
        self.obj_det = self._det_models[0]
        self.device = self.obj_det.device
        self.fp16 = fp16
        self.stride = self.obj_det.stride
        self.imgsz = transform_sz(imgsz)
        self.infsz = transform_sz(imgsz) if infsz is None else transform_sz(infsz)
        self.hooks = {}
        self.transform, self.ratio_pad = self.get_transform(imgsz, infsz)

        models_id2name: List[Dict[int, str]] = [m.names for m in self._det_models]
        self.names, self._merge_mappings = _merge_class_names(models_id2name)
        self._n_shared_cols = self.N_FIXED_COLS + len(self.names)
        # Precompute gather index for vectorized merge (see _merge_detections).
        gather_index, self._max_local_cols = _build_ensemble_gather_index(
            self._merge_mappings,
            self.N_FIXED_COLS,
        )
        self.register_buffer("_gather_col_index", gather_index)  # moves to device with model

        if len(self._det_models) > 1:
            LOGGER.info(f"YOLO ensemble: {len(self._det_models)} models, merged classes={self.names}")

    def forward(self, x, profile=False, visualize=False):
        """Run all models on input and merge their detections into one tensor."""
        preds = [m(x, profile, visualize) for m in self._det_models]
        det_tensors = [p[0] if isinstance(p, tuple) else p for p in preds]
        return (det_tensors[0],) if len(det_tensors) == 1 else (self._merge_detections(det_tensors),)

    def _merge_detections(self, det_tensors: List[torch.Tensor]) -> torch.Tensor:
        """Merge per-model detection tensors into one with class columns remapped to shared space.

        Pads each model to (batch, max_dets, max_local_cols), stacks, gathers columns via
        prebuilt index (zeros for missing classes), then flattens to (batch, n_models * max_dets, total_cols).

        Args:
            det_tensors: One tensor per model, each (batch, num_dets, 5 + local_classes).

        Returns:
            Single tensor (batch, total_dets, 5 + shared_classes).
        """
        batch, n_models = det_tensors[0].shape[0], len(det_tensors)
        max_dets = max(det.shape[1] for det in det_tensors)
        n_shared_cols = self._n_shared_cols

        # Pad to (batch, max_dets, max_local_cols); trailing zeros + missing-class column yield 0.
        padded = [
            torch.nn.functional.pad(det, (0, self._max_local_cols - det.shape[2], 0, max_dets - det.shape[1]), value=0)
            for det in det_tensors
        ]
        stacked = torch.stack(padded, dim=1)  # (B, n_models, max_dets, max_local_cols)

        gather_index = (
            self._gather_col_index.unsqueeze(0)  # (1, n_models, n_shared_cols)
            .unsqueeze(2)  # (1, n_models, 1, n_shared_cols)
            .expand(batch, n_models, max_dets, n_shared_cols)  # (B, n_models, max_dets, n_shared_cols)
        )
        out = torch.gather(stacked, dim=3, index=gather_index)
        return out.reshape(batch, n_models * max_dets, n_shared_cols)

    def register_preprocessing_hook(self):
        """Register hooks to convert uint8 to fp16/fp32 and scale by 1/255 before forward pass."""
        if "preprocessing" in self.hooks:
            return
        self.hooks["preprocessing"] = self.register_forward_pre_hook(self._preprocessing_hook)

    def register_postprocessing_hook(self):
        """Register hooks to convert half to float precision after forward pass."""
        if "postprocessing" in self.hooks:
            return
        self.hooks["postprocessing"] = self.register_forward_hook(self._postprocessing_hook)

    def register_io_hooks(self):
        """Register hooks for input and output processing."""
        self.register_preprocessing_hook()
        self.register_postprocessing_hook()

    def remove_hooks(self):
        """Remove hooks."""
        for _, hook in self.hooks.items():
            hook.remove()
        self.hooks.clear()

    @staticmethod
    def get_transform(imgsz: Tuple[int, int], infsz: Optional[Tuple[int, int]] = None):
        """Get the transformation to be applied to the image. Padding and/or resize, if needed."""
        if imgsz == infsz or infsz is None:
            return None, None

        pad_left, pad_right, pad_top, pad_bottom = YOLO.get_padding_for_aspect_ratio(imgsz, infsz)
        ratio = max(imgsz[0] / infsz[0], imgsz[1] / infsz[1])
        ratio_pad = [[1 / ratio], [pad_left, pad_top]]  # [[gain], [pad_x, pad_y]] for scale_boxes
        transform = transforms.Compose([])
        if ratio != 1:
            transform.transforms.extend(
                [
                    transforms.Resize(
                        (
                            infsz[0] - pad_top - pad_bottom,
                            infsz[1] - pad_left - pad_right,
                        ),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=False,
                    )
                ]
            )
        if pad_left != 0 or pad_right != 0 or pad_top != 0 or pad_bottom != 0:
            transform.transforms.extend(
                [
                    transforms.Pad(
                        padding=(pad_left, pad_top, pad_right, pad_bottom),
                        fill=0,
                        padding_mode="constant",
                    )
                ]
            )
        return transform, ratio_pad

    @staticmethod
    def get_padding_for_aspect_ratio(imgsz, infsz):
        """
        Calculates padding (left, right, top, bottom) needed after resizing
        while preserving aspect ratio to fit exactly into (h_out, w_out).
        """
        h_in, w_in = imgsz
        h_out, w_out = infsz

        scale = min(h_out / h_in, w_out / w_in)
        new_h = int(h_in * scale)
        new_w = int(w_in * scale)

        pad_h = h_out - new_h
        pad_w = w_out - new_w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return pad_left, pad_right, pad_top, pad_bottom

    @staticmethod
    def _preprocessing_hook(module, inputs):
        """Add preprocessing operations to be part of the model."""

        def _preprocess(x: torch.Tensor):
            if len(x.shape) < 1:
                return x
            if module.transform is not None:
                x = x.float()
                x = module.transform(x)
            x = x.half() if module.fp16 else x.float()
            x = x / 255  # 0-255 to 0.0-1.0
            return x

        return tuple(_preprocess(inp) for inp in inputs)

    @staticmethod
    def _postprocessing_hook(module, _inputs, detections):
        """Convert outputs to float (if needed) and scale back boxes if transform was applied."""

        def _to_float(x):
            return x.float() if module.fp16 else x

        # Scale back boxes if transform was applied
        if module.ratio_pad is not None:
            detections = (
                scale_boxes(module.infsz, detections[0], module.imgsz, ratio_pad=module.ratio_pad),
            ) + detections[1:]

        # Convert first item (detection outputs) to float if needed
        detections = (_to_float(detections[0]),) + detections[1:]

        return detections

    def prepare_for_export(self, dynamic: bool = False):
        """Prepare model(s) for export."""
        LOGGER.info(f"✨ Preparing {self.__class__.__name__} for export...")
        for m in self._det_models:
            m.prepare_for_export(dynamic)


class AHOY(YOLO):
    """Base class for AHOY models that extends YOLO with horizon detection.

    AHOY: **A** **H**orizon and **O**bject detection **Y**OLO-based model.
    Inherits preprocessing, hook management, and transform logic from YOLO.
    """

    def __new__(cls, hor_det_weights: str, **kwargs):
        """Create the appropriate AHOY model instance based on the model path.

        Args:
            hor_det_weights: Path to the horizon detection model file.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            An instance of either AHOYv1 or AHOYv2 based on the model path.
        """

        if is_obb_weights(get_weights_path(hor_det_weights)):
            return super().__new__(AHOYv2)
        return super().__new__(AHOYv1)

    def __init__(
        self,
        obj_det_weights: str,
        hor_det_weights: str,
        device: Union[str, torch.device] = "",  # automatically select device
        fp16: bool = False,
        fuse: bool = True,  # fuse conv and bn layers
        imgsz: int | Tuple[int, int] = 640,
        infsz: int | Tuple[int, int] | None = None,
    ):
        """Initialize AHOY model with object detection and horizon detection.

        Args:
            obj_det_weights: Path to object detection model weights
            hor_det_weights: Path to horizon detection model weights
            device: Device to run models on (automatically selected if None)
            fp16: Use half precision (fp16)
            fuse: Fuse conv and batch norm layers
            imgsz: Input image size (height, width)
            infsz: Inference size (height, width). If different from imgsz,
                   the model will apply resize/padding transformations
        """
        # Initialize parent YOLO class with object detection weights
        super().__init__(
            weights=obj_det_weights,
            device=device,
            fp16=fp16,
            fuse=fuse,
            imgsz=imgsz,
            infsz=infsz,
        )

        # Load horizon detection model
        self.hor_det_weights = hor_det_weights  # saved in onnx model metadata
        self.hor_det = self.load_hor_det(get_weights_path(self.hor_det_weights), device=device, fp16=fp16, fuse=fuse)

        LOGGER.debug(
            f"Object detection model info: "
            f"model.type={type(self.obj_det.model).__name__}, "
            f"save={self.obj_det.save}, "
            f"stride={self.obj_det.stride}"
        )
        LOGGER.debug(
            f"Horizon detection model info: "
            f"model.type={type(self.hor_det.model).__name__}, "
            f"save={self.hor_det.save}, "
            f"stride={self.hor_det.stride}"
        )

    def load_hor_det(
        self, hor_det_weights: str, device: Union[str, torch.device] = None, fp16: bool = False, fuse: bool = True
    ):
        """Load horizon detection model."""
        raise NotImplementedError("Subclasses should implement this method")

    def forward(self, x, profile=False, visualize=False):
        """Forward pass through models."""
        raise NotImplementedError("Subclasses should implement this method")

    @staticmethod
    def _postprocessing_hook(module, inputs, outputs):
        """Convert outputs to float (if needed) and apply softmax to logits."""
        raise NotImplementedError("Subclasses should implement this method")

    def prepare_for_export(self, dynamic: bool = False):
        """Prepare both object detection and horizon detection models for export."""
        LOGGER.info(f"✨ Preparing {self.obj_det.__class__.__name__} for export...")
        self.obj_det.prepare_for_export(dynamic)
        LOGGER.info(f"✨ Preparing {self.hor_det.__class__.__name__} for export...")
        self.hor_det.prepare_for_export(dynamic)


class AHOYv1(AHOY):
    """A H-orizon O-bject detection Y-OLOv5 (object detection with yolov5)."""

    def load_hor_det(
        self, hor_det_weights: str, device: Union[str, torch.device] = None, fp16: bool = False, fuse: bool = True
    ):
        """Load horizon detection model."""
        return HorizonModel(hor_det_weights, device=device, fp16=fp16, fuse=fuse)

    def forward(self, x, profile=False, visualize=False):
        """Forward pass through models."""
        objects = self.obj_det(x, profile, visualize)
        pitch, theta = self.hor_det(x, profile, visualize)
        return objects, pitch, theta

    @staticmethod
    def _postprocessing_hook(module, inputs, outputs):
        """Convert outputs to float (if needed) and apply softmax to logits."""

        def _to_float(x):
            return x.float() if module.fp16 else x

        # ahoy outputs: (tuple(Tensor, ...), Tensor, Tensor)
        first_tuple, second_item, third_item = outputs

        # Scale back boxes if transform was applied
        if module.ratio_pad is not None:
            first_tuple = (
                scale_boxes(module.infsz, first_tuple[0], module.imgsz, ratio_pad=module.ratio_pad),
            ) + first_tuple[1:]

        # Only convert the first item of the first tuple (the detection outputs)
        first_tuple = (_to_float(first_tuple[0]),) + first_tuple[1:]

        # second and third items are classification logits
        second_item = second_item.softmax(-1)  # batch dim
        third_item = third_item.softmax(-1)  # batch dim

        # Reconstruct the overall output
        return (first_tuple, _to_float(second_item), _to_float(third_item))


class AHOYv2(AHOY):
    """A H-orizon O-bject detection Y-OLOv5 (object detection with yolov5)."""

    def load_hor_det(
        self, hor_det_weights: str, device: Union[str, torch.device] = None, fp16: bool = False, fuse: bool = True
    ):
        """Load horizon detection model."""
        return OBBModel(hor_det_weights, device=device, fp16=fp16, fuse=fuse)

    def forward(self, x, profile=False, visualize=False):
        """Forward pass through models."""
        objects = self.obj_det(x, profile, visualize)
        horizons = self.hor_det(x)
        return objects, horizons

    @staticmethod
    def _postprocessing_hook(module, inputs, outputs):
        """Convert outputs to float (if needed) and apply softmax to logits."""

        def _to_float(x):
            return x.float() if module.fp16 else x

        # first_tuple: (tuple(Tensor, ...), Tensor)
        # second_tuple: Tensor
        first_tuple, second_item = outputs

        # transpose to (batch_size, num_boxes, num_classes)
        second_item = second_item.transpose(1, 2)

        # Scale back boxes if transform was applied
        if module.ratio_pad is not None:
            first_tuple = (
                scale_boxes(module.infsz, first_tuple[0], module.imgsz, ratio_pad=module.ratio_pad),
            ) + first_tuple[1:]
            second_item = scale_boxes(
                module.infsz, second_item, module.imgsz, ratio_pad=module.ratio_pad, xywh=True, clip=False
            )

        # Only convert the first item of the first tuple (the detection outputs)
        first_tuple = (_to_float(first_tuple[0]),) + first_tuple[1:]
        second_item = _to_float(second_item)

        return (first_tuple, second_item)


class DAN(nn.Module):
    """
    Day
    And
    Night

    Two models in one, ideally one for day and one for night.
    """

    # Ensemble of models
    def __init__(
        self,
        model_a: AHOY,
        model_b: AHOY,
    ):
        super().__init__()
        # check if both models are in same device
        if model_a.device != model_b.device:
            raise ValueError("Both models must be on the same device")
        self.device = model_a.device

        # check if both models are in fp16
        if model_a.fp16 != model_b.fp16:
            raise ValueError("Both models must be in the same precision")
        self.fp16 = model_a.fp16

        self.model_a = model_a
        self.model_b = model_b

    def forward(self, x_1, x_2, profile=False, visualize=False):
        """Forward pass through models."""
        out_a = self.model_a(x_1, profile, visualize)
        out_b = self.model_b(x_2, profile, visualize)
        return out_a, out_b

    def register_io_hooks(self):
        """Register hooks for input and output processing."""
        self.model_a.register_io_hooks()
        self.model_b.register_io_hooks()


def _find_cutoff(model):
    """Find cutoff layer for classification heads."""
    for i, m in enumerate(model.model):
        if m.type == "models.common.SPPF":
            return i - 1
    raise ValueError("Could not find cutoff layer for classification heads.")


def _get_classification_heads(model, cutoff, nc_pitch, nc_theta):
    """
    Get classification heads.

    Similar to method found in
    `models.yolo.ClassificationModel._from_detection_model`
    """

    # get number of input channels for classification heads
    m = model.model[cutoff + 1]  # layer after cutoff
    ch = m.conv.in_channels if hasattr(m, "conv") else m.cv1.conv.in_channels  # ch into module

    # define classification heads
    c_pitch = Classify(ch, nc_pitch)
    c_pitch.i, c_pitch.f, c_pitch.type = "c_pitch", cutoff, "models.common.Classify"
    c_theta = Classify(ch, nc_theta)
    c_theta.i, c_theta.f, c_theta.type = "c_theta", cutoff, "models.common.Classify"

    return c_pitch, c_theta


def _merge_class_names(
    models_id2name: Sequence[Dict[int, str]],
) -> Tuple[Dict[int, str], List[Dict[int, int]]]:
    """Build a shared class vocabulary from per-model class dicts, deduplicating by name (case-insensitive).

    First occurrence of each class name wins; order follows the order models are processed.

    Args:
        models_id2name: One dict per model, each mapping local class index → class name.

    Returns:
        shared_id2name: Shared index → class name (e.g. {0: "person", 1: "car", ...}).
        mappings: One dict per model: local class index → shared class index.
    """
    shared: List[str] = []
    seen: Dict[str, int] = {}  # lowercase name → shared index
    for id2name in models_id2name:
        for name in id2name.values():
            key = name.lower()
            if key not in seen:
                seen[key] = len(shared)
                shared.append(name)

    mappings = [{local_idx: seen[name.lower()] for local_idx, name in id2name.items()} for id2name in models_id2name]
    return dict(enumerate(shared)), mappings


def _build_ensemble_gather_index(
    merge_mappings: List[Dict[int, int]],
    n_fixed_cols: int,
) -> Tuple[torch.Tensor, int]:
    """Build gather index to remap each model's output columns into the shared class space.

    Fixed columns (x1, y1, x2, y2, conf) pass through; class columns use per-model local→shared
    mapping; missing classes read from a zero-filled column.

    Args:
        merge_mappings: Per-model dicts mapping local class index → shared class index.
        n_fixed_cols: Number of fixed columns before class scores (e.g. 5).

    Returns:
        index: Shape (n_models, n_fixed_cols + n_shared_classes).
        max_local_cols: Max model width + 1 (extra column is zero-filled for missing classes).
    """
    all_shared_indices = chain.from_iterable(m.values() for m in merge_mappings)
    n_shared = 1 + max(all_shared_indices, default=-1)
    max_local_cols = 1 + max(n_fixed_cols + len(m) for m in merge_mappings)
    missing_class_offset = (
        max_local_cols - 1 - n_fixed_cols
    )  # column offset when model has no class for that shared index

    index_rows = []
    for mapping in merge_mappings:
        shared_to_local = {v: k for k, v in mapping.items()}
        row = list(range(n_fixed_cols)) + [
            n_fixed_cols + shared_to_local.get(shared_class_idx, missing_class_offset)
            for shared_class_idx in range(n_shared)
        ]
        index_rows.append(row)

    return torch.tensor(index_rows, dtype=torch.long), max_local_cols
