from pdi.src.core.struct_element import StructElement
from pdi.src.core.image import Image
import numpy as np


class Morphology:
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

    @classmethod
    def opening(cls, img: Image, struct_element: StructElement) -> np.ndarray:
        img_h, img_w = img.shape
        erosion_matrix = Morphology.erosion(img, struct_element)

        erosion_img = Image(erosion_matrix, "er")

        dilation_matrix = Morphology.dilatation(erosion_img, struct_element)

        return dilation_matrix

    @classmethod
    def closing(cls, img: Image, struct_element: StructElement) -> np.ndarray:
        img_h, img_w = img.shape
        dilation_matrix = Morphology.dilatation(img, struct_element)

        dilation_img = Image(dilation_matrix, "er")

        erosion_matrix = Morphology.erosion(dilation_img, struct_element)

        return erosion_matrix
