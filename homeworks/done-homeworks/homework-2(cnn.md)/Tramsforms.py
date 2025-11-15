from PIL.Image import Image, BILINEAR
from numpy import random as rand
import numpy as np
import torch
from torch import Tensor
from typing import Tuple, List

class BaseTransform:
    def __init__(self, p: float = 1.0):
        self.p = p
        self.applied = False

    def __call__(self, image: Image) -> Image:
        if rand.rand() < self.p:
            self.applied = True
            return self.apply(image)
        return image

    def apply(self, image: Image) -> Image:
        return image
    
    def reset(self) -> None:
        self.applied = False
    
class RandomCrop(BaseTransform):
    def __init__(self, size: Tuple[int], p: float = 1.0):
        super().__init__(p)
        self.size = size

    def apply(self, image: Image) -> Image:
        width, height = image.size
        new_width, new_height = self.size
        left = rand.randint(0, width - new_width)
        top = rand.randint(0, height - new_height)
        return image.crop((left, top, left + new_width, top + new_height))

class RandomRotate(BaseTransform):
    def __init__(self, degree: int, p: float = 1.0):
        super().__init__(p)
        self.degrees = degree

    def apply(self, image: Image) -> Image:
        corner = rand.uniform(-self.degrees, self.degrees)
        return image.rotate(corner)
    
class RandomZoom(BaseTransform):
    def __init__(self, scale: Tuple[float], p: float = 1.0):
        super().__init__(p)
        self.scale = scale

    def apply(self, image: Image) -> Image:
        width, height = image.size
        scale = rand.uniform(*self.scale)
        new_width, new_height = int(width * scale), int(height * scale)
        image_resized = image.resize((new_width, new_height), BILINEAR)
        left = max(0, (new_width - width) // 2)
        top = max(0, (new_height - height) // 2)
        right = left + width
        bottom = top + height
        return image_resized.crop((left, top, right, bottom))
    
class ToTensor(BaseTransform):
    def __init__(self):
        super().__init__(1.0)
    def apply(self, image: Image) -> Tensor:
        image = np.array(image).astype(np.float32) / 255.0
        if image.ndim == 2:
            image = image[np.newaxis, :, :]
        else:
            image = image.transpose(2, 0, 1)
        return torch.tensor(image)
    
class Compose(BaseTransform):
    def __init__(self, transforms: List[BaseTransform]):
        super().__init__(1.0)
        self.transforms = transforms

    def apply(self, image: Image):
        for transform in self.transforms:
            image = transform(image)
            transform.applied = True
        return image
    
    def reset(self):
        for transform in self.transforms:
            transform.reset()