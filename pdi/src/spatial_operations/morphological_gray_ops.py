from pdi.src.core.image import Image
from pdi.src.core.morphology.grayscale import GrayScaleMorphology
from pdi.src.core.morphology.grayscale import StructElement

import numpy as np
from typing import Tuple, List


class MorphologicalGrayOps:
    def __init__(self) -> None:
        pass

    @classmethod
    def opening(
        cls, img_matrix: np.ndarray, img_shape: Tuple[int, int], kernel: StructElement
    ):
        erosion = GrayScaleMorphology.erosion(img_matrix, img_shape, kernel)

        result = GrayScaleMorphology.dilatation(erosion, img_shape, kernel)

        return result

    @classmethod
    def closing(
        cls, img_matrix: np.ndarray, img_shape: Tuple[int, int], kernel: StructElement
    ):
        dilatation = GrayScaleMorphology.dilatation(img_matrix, img_shape, kernel)

        result = GrayScaleMorphology.erosion(dilatation, img_shape, kernel)

        return result

    @classmethod
    def morphological_gradient(
        cls, img_matrix: np.ndarray, img_shape: Tuple[int, int], kernel: StructElement
    ):
        dilatation = GrayScaleMorphology.dilatation(img_matrix, img_shape, kernel)
        erosion = GrayScaleMorphology.erosion(img_matrix, img_shape, kernel)

        result = dilatation - erosion

        return result

    @classmethod
    def top_hat_transform(
        cls, img_matrix: np.ndarray, img_shape: Tuple[int, int], kernel: StructElement
    ):
        opening = cls.opening(img_matrix, img_shape, kernel)
        result = img_matrix - opening

        return result

    @classmethod
    def bottom_hat_transform(
        cls, img_matrix: np.ndarray, img_shape: Tuple[int, int], kernel: StructElement
    ):
        closing = cls.closing(img_matrix, img_shape, kernel)
        result = img_matrix - closing

        return result

    @classmethod
    def granulometry(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        kernel_sequence: List[StructElement],
    ) -> List[float]:

        reference_sum = img_matrix.sum()
        curve = []

        for kernel in kernel_sequence:
            opened = cls.opening(img_matrix, img_shape, kernel)
            current_sum = opened.sum()
            curve.append((reference_sum - current_sum) / reference_sum * 100)

        return curve

    @classmethod
    def alternate_sequential_filter(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        kernel_sequence: List[StructElement],
    ) -> np.ndarray:

        result = img_matrix.copy()

        for kernel in kernel_sequence:
            result = cls.opening(result, img_shape, kernel)
            result = cls.closing(result, img_shape, kernel)

        return result
