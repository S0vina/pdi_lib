from pdi.src.core.struct_element import StructElement
from pdi.src.core.image import Image
import numpy as np


class Morphology:
    def __init__(self) -> None:
        pass

    @classmethod
    def erode(cls, img: Image, struct_element: StructElement) -> np.ndarray:
        img_h, img_w = img.shape
        result_matrix = img.data.copy()

        for i in range(img_h):
            for j in range(img_w):
                if not struct_element.it_fits(i, j, img.shape, img.data):
                    result_matrix[i][j] = 0

                else:
                    result_matrix[i][j] = img.data[i][j]

        return result_matrix

    @classmethod
    def dilation(cls):
        pass
