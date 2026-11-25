from pdi.src.core.morphology.morphology import Morphology
from pdi.src.core.morphology.struct_element import StructElement

import numpy as np
from typing import Tuple, List


class MorphologicalGrayOps:
    """
    Operações morfológicas de alto nível para imagens grayscale [0, 255].
    Delega para Morphology (erosion/dilation via min/max).
    """

    @classmethod
    def opening(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        kernel: StructElement,
    ) -> np.ndarray:
        eroded = Morphology.erosion(img_matrix, img_shape, kernel)
        return Morphology.dilation(eroded, img_shape, kernel)

    @classmethod
    def closing(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        kernel: StructElement,
    ) -> np.ndarray:
        dilated = Morphology.dilation(img_matrix, img_shape, kernel)
        return Morphology.erosion(dilated, img_shape, kernel)

    @classmethod
    def morphological_gradient(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        kernel: StructElement,
    ) -> np.ndarray:
        dilated = Morphology.dilation(img_matrix, img_shape, kernel)
        eroded = Morphology.erosion(img_matrix, img_shape, kernel)
        return dilated.astype(np.int16) - eroded.astype(np.int16)

    @classmethod
    def top_hat_transform(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        kernel: StructElement,
    ) -> np.ndarray:
        """White top-hat: img - opening(img). Realça picos claros."""
        opening = cls.opening(img_matrix, img_shape, kernel)
        # opening <= img_matrix, subtração segura em uint8
        return img_matrix - opening

    @classmethod
    def bottom_hat_transform(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        kernel: StructElement,
    ) -> np.ndarray:
        """Black top-hat: closing(img) - img. Realça vales escuros."""
        closing = cls.closing(img_matrix, img_shape, kernel)
        # closing >= img_matrix, subtração segura em uint8
        return closing - img_matrix

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
