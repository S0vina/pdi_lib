from typing import List, Tuple
import numpy as np

from .struct_element import StructElement


class Morphology:
    """
    Morfologia unificada para imagens binárias {0, 255} e grayscale [0, 255].

    Convenção: objeto = 255, fundo = 0.
    Padding:   zero-padding — pixels fora dos limites são tratados como 0 (fundo).

    erosion:  min local sob o elemento estruturante
    dilation: max local sob o elemento estruturante
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

    @classmethod
    def hit_or_miss(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        struct_elements: List[StructElement],
    ) -> np.ndarray:
        """
        Detecta padrões exatos com dois SEs disjuntos.
        Entrada e saída em {0, 255}.
        """
        if len(struct_elements) != 2:
            raise ValueError(
                f"hit_or_miss requires exactly 2 structuring elements, got {len(struct_elements)}"
            )

        kernel_one, kernel_two = struct_elements

        if np.any(np.logical_and(kernel_one.array, kernel_two.array)):
            raise ValueError("B1 and B2 must be disjoint structuring elements")

        img_bin = (img_matrix > 0).astype(np.uint8)
        not_img  = (1 - img_bin).astype(np.uint8)

        first_match  = cls.erosion(img_bin * 255, img_shape, kernel_one)
        second_match = cls.erosion(not_img * 255, img_shape, kernel_two)

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
        """
        Hit-or-miss com SE único — complemento gerado automaticamente.
        Entrada e saída em {0, 255}.
        """
        img_bin = (img_matrix > 0).astype(np.uint8)
        not_img  = (1 - img_bin).astype(np.uint8)

        kernel_two_data = (1 - struct_element.array.astype(np.uint8)).astype(bool)
        kh, kw = struct_element.shape
        kernel_two = StructElement(kernel_two_data, None, None, kh, kw)

        first_match  = cls.erosion(img_bin * 255, img_shape, struct_element)
        second_match = cls.erosion(not_img * 255, img_shape, kernel_two)

        result = np.bitwise_and(
            (first_match > 0).astype(np.uint8),
            (second_match > 0).astype(np.uint8),
        )
        return (result * 255).astype(np.uint8)
