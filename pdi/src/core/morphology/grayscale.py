import numpy as np
from typing import Tuple
from .struct_element import StructElement


class GrayScaleMorphology(StructElement):
    def __init__(self, array, mt, mn, h: int, w: int) -> None:
        super().__init__(array, mt, mn, h, w)

    @classmethod
    def dilatation(
        cls, img_matrix: np.ndarray, img_shape: Tuple[int, int], kernel: StructElement
    ):
        img_h, img_w = img_shape
        result = img_matrix.copy()

        for i in range(img_h):
            for j in range(img_w):
                flat_roi = kernel.get_list(i, j, img_matrix)

                result[i][j] = max(flat_roi)

        return result

    @classmethod
    def erosion(
        cls, img_matrix: np.ndarray, img_shape: Tuple[int, int], kernel: StructElement
    ):
        img_h, img_w = img_shape
        result = img_matrix.copy()

        for i in range(img_h):
            for j in range(img_w):
                flat_roi = kernel.get_list(i, j, img_matrix)
                result[i][j] = min(flat_roi)

        return result
