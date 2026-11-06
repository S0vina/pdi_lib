from typing import List, Tuple
from .struct_element import StructElement
import numpy as np


class BinaryMorphology:
    def __init__(self) -> None:
        pass

    @classmethod
    def erosion(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        struct_element: StructElement,
    ) -> np.ndarray:
        img_h, img_w = img_shape
        result_matrix = img_matrix.copy()

        for i in range(img_h):
            for j in range(img_w):
                if not struct_element._fits(i, j, img_matrix, img_shape):
                    result_matrix[i][j] = 255

        return result_matrix

    @classmethod
    def dilatation(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        struct_element: StructElement,
    ) -> np.ndarray:
        img_h, img_w = img_shape
        result_matrix = img_matrix.copy()

        for i in range(img_h):
            for j in range(img_w):
                if struct_element._hits(i, j, img_matrix, img_shape):
                    result_matrix[i][j] = 0

        return result_matrix

    @classmethod
    def hit_or_miss(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        struct_elements: List[StructElement],
    ):

        if len(struct_elements) != 2:
            raise ValueError(
                f"hit_or_miss requires exactly 2 structuring elements, got {len(struct_elements)}"
            )

        not_img = 1 - img_matrix
        kernel_one = struct_elements[0]
        kernel_two = struct_elements[1]

        if np.any(np.logical_and(kernel_one.array, kernel_two.array)):
            raise ValueError("B1 and B2 must be disjoint structuring elements")

        first_match = BinaryMorphology.erosion(img_matrix, img_shape, kernel_one)
        second_match = BinaryMorphology.erosion(not_img, img_shape, kernel_two)

        return np.logical_and(first_match, second_match).astype(np.uint8)

    @classmethod
    def hit_or_miss_simplified(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        struct_element: StructElement,
    ):
        not_img = 1 - img_matrix
        kernel_one = struct_element
        kernel_two_data = 1 - kernel_one.array
        kernel_two_h, kernel_two_w = kernel_one.shape
        kernel_two = StructElement(
            kernel_two_data, None, None, kernel_two_h, kernel_two_w
        )

        first_match = BinaryMorphology.erosion(img_matrix, img_shape, kernel_one)
        second_match = BinaryMorphology.erosion(not_img, img_shape, kernel_two)

        return np.logical_and(first_match, second_match).astype(np.uint8)
