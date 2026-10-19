from typing import Tuple

from src.plugins.Image import Image
from src.plugins.Mask import Mask
import numpy as np

class LinearSpatialFilter():
    def __init__(self) -> None:
        pass
    
    @classmethod
    def correlation(cls, img: Image, mask: Mask) -> np.ndarray:
        img_h, img_w = img.shape
        
        for i in range(img_h):
            for j in range(img_w):
                if LinearSpatialFilter.it_fits(i, j, mask.shape, img.shape)
                    for(k in range())
    
    @classmethod
    def convolution(cls, img: Image, f: Mask) -> np.ndarray:
        pass

    @classmethod
    def it_fits(cls, i: int, j: int, mask_t: Tuple[int, int], img_t: Tuple[int, int]) -> bool:
        mask_h, mask_w = mask_t
        
        # testannto se tem espaço em cima ou em baixo
        if (mask_h / 2) - i < 0:
            return False

        if (mask_h / 2) + i > (img_t[0] - 1):
            return False
        
        if (mask_w / 2) - j < 0: 
            return False

        if (mask_w / 2) + j > (img_t[1] - 1):
            return False

        return True
        
        



