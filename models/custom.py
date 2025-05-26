from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from ultralytics import YOLO

from models.common import Classify, DetectMultiBackend
from models.experimental import attempt_load
from models.yolo import BaseModel, DetectionModel
from utils.general import LOGGER
from utils.plots import feature_visualization
from utils.torch_utils import select_device


class OBBModel(BaseModel):
    """Wrapper around yolo-OBB Model."""

    def __init__(
        self,
        weights: str = "yolov11n-obb.pt",
        device: Union[str, torch.device] = None,  # automatically select device
        fp16: bool = False,
        fuse: bool = False,
    ):
        super().__init__()
        self.device = select_device(device)
        self.fp16 = fp16

        if Path(weights).is_file() or weights.endswith(".pt"):
            model = YOLO(model=weights)
            if fuse:
                model.fuse()
            model = model.model
            stride = model.stride
            LOGGER.info(f"Loaded weights from {weights}")
        else:
            raise ValueError("model must be a path to a .pt file")

        self.model = model.model
        self.model.to(self.device)
        self.model.half() if fp16 else self.model.float()
        self.stride = stride
        self.save = model.save

class HorizonModel(BaseModel):
    """YOLOv5 backbone + classification heads for pitch and theta."""

    def __init__(
        self,
        weights: str = "yolov5n.pt",
        nc_pitch: int = 500,
        nc_theta: int = 500,
        device: Union[str, torch.device] = None,  # automatically select device
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
        self.device = select_device(device)
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
      
        self.model.half() if fp16 else self.model.float()
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


class ObjectsModel(BaseModel):
    """Wrapper around YOLOv5 DetectionModel."""

    def __init__(
        self,
        weights: str = "yolov5n.pt",
        device: Union[str, torch.device] = None,  # automatically select device
        fp16: bool = False,
        fuse: bool = False,
    ):
        super().__init__()
        self.device = select_device(device)
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
        self.model.half() if fp16 else self.model.float()
        self.stride = stride
        self.names = names
        self.save = model.save



class AHOY(nn.Module):
    """A H-orizon O-bject detection Y-OLOv5."""

    # Ensemble of models
    def __init__(
        self,
        obj_det_weigths: str,
        hor_det_weights: str,
        device: Union[str, torch.device] = None,  # automatically select device
        fp16: bool = False,
        fuse: bool = True,  # fuse conv and bn layers
        imgsz: list = [640,640],
        infsz: list = [640,640],
        inplace: bool = True,  # inplace modification of models
    ):
        super().__init__()
        self.obj_det = ObjectsModel(obj_det_weigths, device=device, fp16=fp16, fuse=fuse)
        self.hor_det = HorizonModel(hor_det_weights, device=device, fp16=fp16, fuse=fuse)
        self.device = select_device(device)
        self.fp16 = fp16
        self.stride = self.obj_det.stride
        self.names = self.obj_det.names
        self.imgsz = imgsz
        self.infsz = infsz

        # keep track of hooks
        self.hooks = {}

        # Scaling in Padding Preprocessing
        self.transform = self.get_transform(imgsz, infsz)

    def forward(self, x, profile=False, visualize=False):
        """Forward pass through models."""
        objects = self.obj_det(x, profile, visualize)
        pitch, theta = self.hor_det(x, profile, visualize)
        return objects, pitch, theta

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
    def get_transform(imgsz, infsz):
        """Get the transformation to be applied to the image. Padding and/or resize, if needed."""
        if imgsz == infsz:
            return None

        pad_left, pad_right, pad_top, pad_bottom = AHOY.get_padding_for_aspect_ratio(imgsz, infsz)
        ratio = max(imgsz[0] / infsz[0], imgsz[1] / infsz[1])
        transform = transforms.Compose([])
        if ratio != 1:
            transform.transforms.extend(
                [transforms.Resize((infsz[0]-pad_top-pad_bottom, infsz[1]-pad_left-pad_right), interpolation=transforms.InterpolationMode.NEAREST, antialias=False)]
            )
        if pad_left != 0 or pad_right != 0 or pad_top != 0 or pad_bottom != 0:
            transform.transforms.extend(
                [transforms.Pad(padding=(pad_left, pad_top, pad_right, pad_bottom), fill = 0, padding_mode="constant")]
            )
        return transform

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

        def _preprocess(x):
            if len(x.shape) <1 :
                return x
            if module.transform is not None:
                x = x.float()
                x = module.transform(x)
            x = x.half() if module.fp16 else x.float()
            x = x / 255.0  # 0-255 to 0.0-1.0
            return x

        return tuple(_preprocess(inp) for inp in inputs)

    @staticmethod
    def _postprocessing_hook(module, inputs, outputs):
        """Convert outputs to float (if needed) and apply softmax to logits."""

        def _to_float(x):
            return x.float() if module.fp16 else x

        # ahoy outputs: (tuple(Tensor, ...), Tensor, Tensor)
        first_tuple, second_item, third_item = outputs

        # Only convert the first item of the first tuple (the detection outputs)
        first_tuple = (_to_float(first_tuple[0]),) + first_tuple[1:]

        # second and third items are classification logits
        second_item = second_item.softmax(-1)  # batch dim
        third_item = third_item.softmax(-1)  # batch dim

        # Reconstruct the overall output
        return (first_tuple, _to_float(second_item), _to_float(third_item))


class AHOYOBB(AHOY):
    """A H-orizon (with OBB) O-bject detection Y-OLOv5 (object detection with yolov5)."""

    # Ensemble of models
    def __init__(
        self,
        obj_det_weigths: str,
        hor_det_weights: str,
        device: Union[str,
                    torch.device] = None,  # automatically select device
        fp16: bool = False,
        fuse: bool = True,  # fuse conv and bn layers
        imgsz: list = [640, 640],
        infsz: list = [640, 640],
        inplace: bool = True,  # inplace modification of models
    ):
        nn.Module.__init__(self) 
        self.obj_det = ObjectsModel(obj_det_weigths, device=device, fp16=fp16, fuse=fuse)
        self.hor_det = OBBModel(hor_det_weights, device=device, fp16=fp16, fuse=fuse)
        self.device = select_device(device)
        self.fp16 = fp16
        self.stride = self.obj_det.stride
        self.names = self.obj_det.names
        self.imgsz = imgsz
        self.infsz = infsz

        # keep track of hooks
        self.hooks = {}
        print("🚀 AHOY model: yolov5 + yolo[v8|11]-obb")
        # Scaling in Padding Preprocessing
        self.transform = self.get_transform(imgsz, infsz)

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

        # ahoy outputs: (tuple(Tensor, ...), Tensor, Tensor)
        first_tuple, second_tuple = outputs

        # Only convert the first item of the first tuple (the detection outputs)
        first_tuple = (_to_float(first_tuple[0]), ) + first_tuple[1:]
        second_tuple = (_to_float(second_tuple[0].transpose(1, 2)), ) 

        return (first_tuple, second_tuple)

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


class Hydra(BaseModel):
    """
    Model with two heads: object detection and horizon detection.
    HydraModel is a wrapper around YOLOv5 DetectionModel.
    In Greek mythology, Hydra is a serpent-like monster with many heads.
    """

    def __init__(
        self,
        weights: str = "yolov5n.pt",
        nc_pitch: int = 500,
        nc_theta: int = 500,
        device: Union[str, torch.device] = None,  # automatically select device
        cutoff: int = None,
        fp16: bool = False,
        task: str = "both",  # "detection", "horizon", "both"
    ):
        """
        NOTE: Not tested!!!
        Multi-task model with object detection and horizon detection.
        backbone --> neck --> detection heads
                 └-> classification heads for pitch and theta

        Args:
            model (DetectionModel): YOLOv5 model
            nc_pitch (int, optional): number of classes for pitch classification. Defaults to 500.
            nc_theta (int, optional): number of classes for theta classification. Defaults to 500.
            device (str, optional): device to run model on. Defaults to ''.
            cutoff (int, optional): cutoff layer for classification heads.
                If not specified, the SPPF layer is used as cutoff. Defaults to None.
            task (str, optional): task to run. Defaults to "both".
                Possible values: "detection", "horizon", "both"
        """
        super().__init__()

        assert weights is not None, "weights must be specified"
        assert isinstance(nc_pitch, int), "nc_pitch must be an integer"
        assert isinstance(nc_theta, int), "nc_theta must be an integer"
        assert isinstance(fp16, bool), "fp16 must be a boolean"
        assert task in [
            "detection",
            "horizon",
            "both",
        ], "task must be one of 'detection', 'horizon', 'both'"

        self.nc_pitch = nc_pitch
        self.nc_theta = nc_theta
        self.device = select_device(device)
        self.fp16 = fp16
        self.task = task

        if Path(weights).is_file() or weights.endswith(".pt"):
            model = attempt_load(weights, device="cpu", fuse=False)
            LOGGER.info(f"Loaded weights from {weights}")
        else:
            raise ValueError("model must be a path to a .pt file")

        if isinstance(model, DetectionModel):
            LOGGER.warning("WARNING ⚠️ converting YOLOv5 DetectionModel to HorizonModel")
            self.cutoff = _find_cutoff(model) if cutoff is None else cutoff
            self._add_classification_heads(model, self.cutoff)  # inplace modification

        self.model = model.model
        self.model.to(self.device)
        self.model.half() if fp16 else self.model.float()
        self.save = model.save
        self.stride = model.stride
        self.nc = model.nc

    def _add_classification_heads(self, model: DetectionModel, cutoff: int):
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend

        c_pitch, c_theta = _get_classification_heads(model, cutoff, self.nc_pitch, self.nc_theta)
        model.save = set(list(model.save + [cutoff]))  # add cutoff to save

        # add classification heads to model
        model.model.add_module(c_pitch.i, c_pitch)
        model.model.add_module(c_theta.i, c_theta)

    def _horizon_once(self, x, profile=False, visualize=False):
        x_pitch, x_theta = None, None
        y, dt = [], []  # outputs
        for m in self.model:
            if isinstance(m.i, int) and m.i > self.cutoff:
                continue
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if m.type == "models.common.Classify" and "pitch" in m.i:
                x_pitch = m(x)
            elif m.type == "models.common.Classify" and "theta" in m.i:
                x_theta = m(x)
            else:  # object detection flow
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
                if visualize:
                    feature_visualization(x, m.type, m.i, save_dir=visualize)

        return (x_pitch, x_theta)

    def _detect_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.type == "models.common.Classify":
                continue
            if profile:
                self._profile_one_layer(m, x, dt)
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

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
            else:  # object detection flow
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
                if visualize:
                    feature_visualization(x, m.type, m.i, save_dir=visualize)

        return (x, x_pitch, x_theta)

    def forward(self, x):
        if self.task == "detection":
            return self._detect_once(x)
        elif self.task == "horizon":
            return self._horizon_once(x)
        else:
            return self._forward_once(x)


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
