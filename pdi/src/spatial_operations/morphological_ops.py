from types import ClassMethodDescriptorType

from pdi.src.core.morphology.struct_element import StructElement
from pdi.src.core.morphology.binary import BinaryMorphology
from pdi.src.core.image import Image
import numpy as np


class MorphologicalOps:
    def __init__(self) -> None:
        pass

    @classmethod
    def opening(cls, img: Image, struct_element: StructElement) -> np.ndarray:
        erosion_matrix = BinaryMorphology.erosion(img.data, img.shape, struct_element)

        dilation_matrix = BinaryMorphology.dilatation(
            erosion_matrix, img.shape, struct_element
        )

        return dilation_matrix

    @classmethod
    def closing(cls, img: Image, struct_element: StructElement) -> np.ndarray:
        dilation_matrix = BinaryMorphology.dilatation(
            img.data, img.shape, struct_element
        )

        erosion_matrix = BinaryMorphology.erosion(
            dilation_matrix, img.shape, struct_element
        )

        return erosion_matrix

    @classmethod
    def boudary_extraction(
        cls, img: Image, struct_element: StructElement
    ) -> np.ndarray:
        erosion_matrix = BinaryMorphology.erosion(img.data, img.shape, struct_element)

        boundary_matrix = img.data - erosion_matrix

        return boundary_matrix

    @classmethod
    def hole_filling(cls, img: Image, struct_element: StructElement) -> np.ndarray:
        not_img = 1 - img.data

        full_matrix = np.zeros_like(img.data)

        full_matrix[0, :] = not_img[0, :]
        full_matrix[-1, :] = not_img[-1, :]
        full_matrix[:, 0] = not_img[:, 0]
        full_matrix[:, -1] = not_img[:, -1]

        while True:
            k_minusone_matrix = np.copy(full_matrix)
            dilated = BinaryMorphology.dilatation(
                full_matrix, img.shape, struct_element
            ).astype(np.uint8)
            full_matrix = np.logical_and(dilated, not_img).astype(np.uint8)
            # no change between dilatations
            if np.array_equal(k_minusone_matrix, full_matrix):
                break

        holes = not_img - full_matrix

        hole_filed = np.logical_or(holes, img.data)
        return hole_filed
