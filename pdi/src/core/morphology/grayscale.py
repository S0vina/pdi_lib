import numpy as np
from typing import Tuple
from .struct_element import StructElement
from .morphology import Morphology


class GrayScaleMorphology:
    """
    Morfologia grayscale. Convenção: objeto = 255, fundo = 0.

    Delega para Morphology (min/max), idêntico à morfologia binária
    mas operando sobre valores contínuos [0, 255].
    """

    @classmethod
    def erosion(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        kernel: StructElement,
    ) -> np.ndarray:
        return Morphology.erosion(img_matrix, img_shape, kernel)

    @classmethod
    def dilatation(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        kernel: StructElement,
    ) -> np.ndarray:
        return Morphology.dilation(img_matrix, img_shape, kernel)
