from pdi.src.core.image import Image
from typing import Tuple
import numpy as np


class IntensityTransforms:
    def __init__(self) -> None:
        pass

    @classmethod
    def threshold(
        cls, img: np.ndarray, img_shape: Tuple[int, int], thresshold: int = 127
    ) -> np.ndarray:
        thr_img = np.zeros_like(img.data)
        rows, colluns = img_shape

        for i in range(rows):
            for j in range(colluns):
                if img[i][j] > thresshold:
                    thr_img[i][j] = 255
                else:
                    thr_img[i][j] = 0

        return thr_img

    @classmethod
    def negative(cls, img: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        neg_img = np.zeros_like(img)
        bigger_intesity = np.iinfo(img.dtype).max
        rows, colluns = img_shape

        for i in range(rows):
            for j in range(colluns):
                neg_img[i][j] = bigger_intesity - 1 - img[i][j]

        return neg_img

    @classmethod
    def logaritmic(
        cls, img: np.ndarray, img_shape: Tuple[int, int], const: int
    ) -> np.ndarray:
        log_img = np.zeros_like(img)
        rows, colums = img_shape

        for i in range(rows):
            for j in range(colums):
                log_img[i][j] = const * np.log(1 + img[i][j])

        return log_img
