from src.plugins.Image import Image
from src.plugins.Mask import Mask
import numpy as np


class NonLinearSpatialFilter:
    def __init__(self) -> None:
        pass

    @classmethod
    def median(cls, img: Image, n: int = 3) -> np.ndarray:
        if n % 2 != 1:
            raise ValueError("Median method should have a quadratic roi")
        img_h, img_w = img.shape
        matriz = np.zeros((img_h, img_w))
        fit_mask = Mask.any_shape(n, n, 0)
        center = int(n / 2)

        for i in range(img_h):
            for j in range(img_w):
                if fit_mask.it_fits(i, j, img.shape):
                    roi = img.data[i : i + n, j : j + n]
                    roi.sort()
                    matriz[i][j] = roi[center][center]

        return matriz

    @classmethod
    def max_filter(cls, img: Image):
        pass

    @classmethod
    def min_filter(cls, img: Image):
        pass
