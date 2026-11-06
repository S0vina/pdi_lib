from pdi.src.core.morphology.struct_element import StructElement
from pdi.src.core.morphology.binary import BinaryMorphology
from pdi.src.core.image import Image
import numpy as np


class MorphologicalOps:
    def __init__(self) -> None:
        pass

    @classmethod
    def opening(cls, img: Image, struct_element: StructElement) -> np.ndarray:
        img_h, img_w = img.shape
        erosion_matrix = BinaryMorphology.erosion(img, struct_element)

        erosion_img = Image(erosion_matrix, "er")

        dilation_matrix = BinaryMorphology.dilatation(erosion_img, struct_element)

        return dilation_matrix

    @classmethod
    def closing(cls, img: Image, struct_element: StructElement) -> np.ndarray:
        img_h, img_w = img.shape
        dilation_matrix = BinaryMorphology.dilatation(img, struct_element)

        dilation_img = Image(dilation_matrix, "er")

        erosion_matrix = BinaryMorphology.erosion(dilation_img, struct_element)

        return erosion_matrix
