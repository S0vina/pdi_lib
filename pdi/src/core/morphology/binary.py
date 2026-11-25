from typing import List, Tuple
from .struct_element import StructElement
from .morphology import Morphology
import numpy as np


class BinaryMorphology:
    """
    Morfologia binária. Convenção: objeto = 255, fundo = 0.

    erosion e dilation delegam para Morphology (min/max), que é equivalente
    às implementações originais _fits/_hits e funciona igualmente para grayscale.

    hit_or_miss normaliza internamente para {0, 1} e retorna {0, 255}.
    """

    @classmethod
    def erosion(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        struct_element: StructElement,
    ) -> np.ndarray:
        return Morphology.erosion(img_matrix, img_shape, struct_element)

    @classmethod
    def dilatation(
        cls,
        img: np.ndarray,
        img_shape: Tuple[int, int],
        struct_element: StructElement,
    ) -> np.ndarray:
        return Morphology.dilation(img, img_shape, struct_element)

    @classmethod
    def hit_or_miss(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        struct_elements: List[StructElement],
    ) -> np.ndarray:
        if len(struct_elements) != 2:
            raise ValueError(
                f"hit_or_miss requires exactly 2 structuring elements, got {len(struct_elements)}"
            )

        kernel_one = struct_elements[0]
        kernel_two = struct_elements[1]

        if np.any(np.logical_and(kernel_one.array, kernel_two.array)):
            raise ValueError("B1 and B2 must be disjoint structuring elements")

        # normaliza para {0, 1} para complemento correto
        img_bin = (img_matrix > 0).astype(np.uint8)
        not_img = 1 - img_bin

        first_match = Morphology.erosion(img_bin * 255, img_shape, kernel_one)
        second_match = Morphology.erosion(not_img * 255, img_shape, kernel_two)

        result = np.bitwise_and(
            (first_match > 0).astype(np.uint8),
            (second_match > 0).astype(np.uint8),
        )
        return (result * 255).astype(np.uint8)

    @classmethod
    def hit_or_miss_simplified(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        struct_element: StructElement,
    ) -> np.ndarray:
        # normaliza para {0, 1} para complemento correto
        img_bin = (img_matrix > 0).astype(np.uint8)
        not_img = 1 - img_bin

        kernel_one = struct_element
        kernel_two_data = (1 - kernel_one.array.astype(np.uint8)).astype(bool)
        kernel_two_h, kernel_two_w = kernel_one.shape
        kernel_two = StructElement(
            kernel_two_data, None, None, kernel_two_h, kernel_two_w
        )

        first_match = Morphology.erosion(img_bin * 255, img_shape, kernel_one)
        second_match = Morphology.erosion(not_img * 255, img_shape, kernel_two)

        result = np.bitwise_and(
            (first_match > 0).astype(np.uint8),
            (second_match > 0).astype(np.uint8),
        )
        return (result * 255).astype(np.uint8)
