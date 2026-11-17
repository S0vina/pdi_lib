from pdi.src.core.mask import Mask
from typing import Tuple
import numpy as np


class LinearSpatialFilter:
    def __init__(self) -> None:
        pass

    @classmethod
    def correlation(
        cls, img_matrix: np.ndarray, img_shape: Tuple[int, int], mask: Mask
    ) -> np.ndarray:
        img_h, img_w = img_shape
        mask_h, mask_w = mask.shape

        result_matrix = np.zeros_like(img_matrix)

        for i in range(img_h):
            for j in range(img_w):
                if mask.it_fits(i, j, img_shape):
                    roi = img_matrix[i : i + mask_h, j : j + mask_w]

                    result_matrix[i][j] = round(
                        min(255, max(0, np.sum(roi * mask.array)))
                    )

        int_matrix = np.clip(result_matrix, 0, 255).astype(np.uint8)
        return int_matrix

    @classmethod
    def convolution(
        cls, img_matrix: np.ndarray, img_shape: Tuple[int, int], mask: Mask
    ) -> np.ndarray:
        img_h, img_w = img_shape

        flipped_mask = Mask.flip(mask)
        f_mask_h, f_mask_w = flipped_mask.shape

        result_matrix = np.zeros((img_h, img_w))

        for i in range(img_h):
            for j in range(img_w):
                result_matrix[i][j] = flipped_mask.dot_product(
                    i, j, img_matrix, img_shape
                )
        int_matrix = np.clip(result_matrix, 0, 255).astype(np.uint8)

        return int_matrix

    @classmethod
    def low_pass(cls, img_matrix: np.ndarray, img_shape: Tuple[int, int]):
        mask = Mask.from_type(Mask.LOW_PASS, Mask.EIGHT_NEIGHBOR)

        return cls.convolution(img_matrix, img_shape, mask)

    @classmethod
    def laplassian(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        neighboor: str = Mask.EIGHT_NEIGHBOR,
    ) -> np.ndarray:
        mask = Mask.from_type(Mask.LAPLASSIAN, neighboor)
        img_h, img_w = img_shape

        instance = np.zeros_like(img_matrix)

        for i in range(img_h):
            for j in range(img_w):
                val = float(img_matrix[i][j]) + float(
                    mask.dot_product(i, j, img_matrix, img_shape)
                )
                instance[i][j] = int(min(255, max(0, val)))

        return instance
