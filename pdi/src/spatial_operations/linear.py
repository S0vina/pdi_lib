from sys import implementation

from pdi.src.core.image import Image
from pdi.src.core.mask import Mask
import numpy as np


class LinearSpatialFilter:
    def __init__(self) -> None:
        pass

    @classmethod
    def correlation(cls, img: Image, mask: Mask) -> np.ndarray:
        img_h, img_w = img.shape
        mask_h, mask_w = mask.shape

        n_img = np.zeros_like(img.data)

        for i in range(img_h):
            for j in range(img_w):
                if mask.it_fits(i, j, img.shape):
                    roi = img.data[i : i + mask_h, j : j + mask_w]

                    n_img[i][j] = round(min(255, max(0, np.sum(roi * mask.array))))

        return n_img

    @classmethod
    def convolution(cls, img: Image, mask: Mask) -> np.ndarray:
        img_h, img_w = img.shape

        flipped_mask = Mask.flip(mask)
        f_mask_h, f_mask_w = flipped_mask.shape

        array = np.zeros((img_h, img_w))

        for i in range(img_h):
            for j in range(img_w):
                array[i][j] = flipped_mask.dot_product(i, j, img.data, img.shape)
                print(array[i][j])
        return array

    @classmethod
    def low_pass(cls, img: Image):
        mask = Mask.from_type(Mask.LOW_PASS, Mask.EIGHT_NEIGHBOR)
        # print(mask.array)

        return cls.convolution(img, mask)

    @classmethod
    def laplassian(cls, img: Image, neighboor: str = Mask.EIGHT_NEIGHBOR) -> np.ndarray:
        mask = Mask.from_type(Mask.LAPLASSIAN, neighboor)
        img_h, img_w = img.shape

        instance = np.zeros_like(img.data)

        for i in range(img_h):
            for j in range(img_w):
                val = float(img.data[i][j]) + float(
                    mask.dot_product(i, j, img.data, img.shape)
                )
                instance[i][j] = int(min(255, max(0, val)))

        return instance
