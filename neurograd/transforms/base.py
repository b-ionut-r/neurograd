from neurograd import Tensor, xp
from neurograd.functions.base import Function
from abc import ABC, abstractmethod
from typing import Union, Literal
from pathlib import Path
from PIL import Image


def load_image(path: Union[str, Path],
               shape: tuple = None,
               mode: str = None,
               chw: bool = True,
               dtype = xp.float32,
               output_type: Literal["array", "tensor"] = "tensor"):
    with Image.open(path) as img:
        if mode is not None:
            img = img.convert(mode)
        if shape is not None:
            img = img.resize(shape[::-1], resample=Image.BILINEAR)  # (W,H)
        arr = xp.array(img)  # uint8 HxWxC or HxW
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if chw:
        arr = xp.transpose(arr, (2, 0, 1))  # C,H,W


class BaseTransform(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def __call__(self, image):
        pass
    def __repr__(self):
        return f"{self.__class__.__name__}()"


class BaseRandomTransform(BaseTransform):
    def __init__(self, p: float):
        self.p = p
    def __call__(self, image: Union[xp.ndarray, Tensor, Path, str],
                 output: Literal["array", "tensor"] = "array") -> Tensor:
        if isinstance(image, (str, Path)):
            image = load_image(image, output_type="array")
        if isinstance(image, Tensor):
            image = image.data
        if xp.random.rand() < self.p:
            return self.apply(image)
        return image
    @abstractmethod
    def apply(self, image: xp.ndarray) -> xp.ndarray:
        pass
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"
