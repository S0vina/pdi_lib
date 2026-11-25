from typing import Tuple
import numpy as np

from .struct_element import StructElement


class Morphology:
    """
    Morfologia unificada para imagens binárias {0, 255} e grayscale [0, 255].

    Convenção: objeto = 255, fundo = 0.
    Padding:   zero-padding — pixels fora dos limites são tratados como 0 (fundo).

    Ambas as operações funcionam identicamente para imagens binárias e grayscale:
      - erosion:  min local sob o elemento estruturante
      - dilation: max local sob o elemento estruturante

    Para imagens binárias {0, 255}:
      min retorna 0 se qualquer vizinho for fundo  → erosão encolhe o objeto
      max retorna 255 se qualquer vizinho for objeto → dilatação expande o objeto
    """

    @classmethod
    def erosion(
        cls,
        img: np.ndarray,
        img_shape: Tuple[int, int],
        kernel: StructElement,
    ) -> np.ndarray:
        """Erosão por mínimo local. Funciona para binária e grayscale."""
        img_h, img_w = img_shape
        kh, kw = kernel.shape
        half_h, half_w = kh // 2, kw // 2
        result = np.zeros_like(img)

        for i in range(img_h):
            for j in range(img_w):
                min_val = 255
                for ki in range(kh):
                    for kj in range(kw):
                        if not kernel.array[ki][kj]:
                            continue
                        r = i + ki - half_h
                        c = j + kj - half_w
                        val = int(img[r][c]) if 0 <= r < img_h and 0 <= c < img_w else 0
                        if val < min_val:
                            min_val = val
                result[i][j] = min_val

        return result

    @classmethod
    def dilation(
        cls,
        img: np.ndarray,
        img_shape: Tuple[int, int],
        kernel: StructElement,
    ) -> np.ndarray:
        """Dilatação por máximo local. Funciona para binária e grayscale."""
        img_h, img_w = img_shape
        kh, kw = kernel.shape
        half_h, half_w = kh // 2, kw // 2
        result = np.zeros_like(img)

        for i in range(img_h):
            for j in range(img_w):
                max_val = 0
                for ki in range(kh):
                    for kj in range(kw):
                        if not kernel.array[ki][kj]:
                            continue
                        r = i + ki - half_h
                        c = j + kj - half_w
                        val = int(img[r][c]) if 0 <= r < img_h and 0 <= c < img_w else 0
                        if val > max_val:
                            max_val = val
                result[i][j] = max_val

        return result
