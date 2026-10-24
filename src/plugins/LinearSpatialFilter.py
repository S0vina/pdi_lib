from src.plugins.Image import Image
from src.plugins.Mask import Mask
import numpy as np


class LinearSpatialFilter:
    def __init__(self) -> None:
        pass

    @classmethod
    def correlation(cls, img: Image, mask: Mask) -> np.ndarray:
        img_h, img_w = img.shape
        mask_h, mask_w = mask.shape

        n_img = np.zeros_like(img.data)
        flip_mask = Mask.flip(mask)

        for i in range(img_h):
            for j in range(img_w):
                if flip_mask.it_fits(i, j, img.shape):
                    roi = img.data[i : i + mask_h, j : j + mask_w]

                    n_img[i][j] = min(255, max(0, np.sum(roi * flip_mask)))

        return n_img

    @classmethod
    def convolution(cls, img: Image, mask: Mask) -> np.ndarray:
        img_h, img_w = img.shape
        mask_h, mask_w = mask.shape

        n_img = np.zeros_like(img.data)

        for i in range(img_h):
            for j in range(img_w):
                if mask.it_fits(i, j, img.shape):
                    roi = img.data[i : i + mask_h, j : j + mask_w]

                    n_img[i][j] = min(255, max(0, np.sum(roi * mask)))

        return n_img
