from .struct_element import StructElement
from pdi.src.core.image import Image
import numpy as np


class BinaryMorphology:
    def __init__(self) -> None:
        pass

    @classmethod
    def erosion(cls, img: Image, struct_element: StructElement) -> np.ndarray:
        img_h, img_w = img.shape
        result_matrix = img.data.copy()

        for i in range(img_h):
            for j in range(img_w):
                if not struct_element.it_fits(i, j, img.shape, img.data):
                    result_matrix[i][j] = 255

        return result_matrix

    @classmethod
    def dilatation(
        cls,
        img: Image,
        struct_element: StructElement,
    ) -> np.ndarray:
        img_h, img_w = img.shape
        result_matrix = img.data.copy()

        for i in range(img_h):
            for j in range(img_w):
                if struct_element.it_fits(i, j, img.shape, img.data, True):
                    result_matrix[i][j] = 0

        return result_matrix
