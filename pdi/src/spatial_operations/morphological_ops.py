from pdi.src.core.morphology.struct_element import StructElement
from pdi.src.core.morphology.morphology import Morphology
from typing import List, Tuple
import numpy as np


class MorphologicalOps:
    """
    Operações morfológicas de alto nível para imagens binárias {0, 255}.
    Todas delegam para Morphology (erosion/dilation via min/max).
    """

    @classmethod
    def opening(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        struct_element: StructElement,
    ) -> np.ndarray:
        eroded = Morphology.erosion(img_matrix, img_shape, struct_element)
        return Morphology.dilation(eroded, img_shape, struct_element)

    @classmethod
    def closing(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        struct_element: StructElement,
    ) -> np.ndarray:
        dilated = Morphology.dilation(img_matrix, img_shape, struct_element)
        return Morphology.erosion(dilated, img_shape, struct_element)

    @classmethod
    def boudary_extraction(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        struct_element: StructElement,
    ) -> np.ndarray:
        eroded = Morphology.erosion(img_matrix, img_shape, struct_element)
        return img_matrix - eroded

    @classmethod
    def hole_filling(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        struct_element: StructElement,
    ) -> np.ndarray:
        not_img = 255 - img_matrix

        x_matrix = np.zeros_like(img_matrix)
        x_matrix[0, :]  = not_img[0, :]
        x_matrix[-1, :] = not_img[-1, :]
        x_matrix[:, 0]  = not_img[:, 0]
        x_matrix[:, -1] = not_img[:, -1]

        while True:
            prev = x_matrix.copy()
            dilated = Morphology.dilation(x_matrix, img_shape, struct_element)
            x_matrix = np.bitwise_and(dilated, not_img)
            if np.array_equal(prev, x_matrix):
                break

        holes = not_img - x_matrix
        return np.bitwise_or(holes, img_matrix)

    @classmethod
    def connected_components_extraction(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        struct_element: StructElement,
    ) -> List[Tuple[int, np.ndarray]]:
        remaining = img_matrix.copy()
        components = []
        label = 1
        img_h, img_w = img_shape

        while True:
            seed = None
            for i in range(img_h):
                for j in range(img_w):
                    if remaining[i][j] == 255:
                        seed = (i, j)
                        break
                if seed:
                    break

            if seed is None:
                break

            D = np.zeros_like(img_matrix)
            D[seed] = 255

            while True:
                D_prev = D.copy()
                dilated = Morphology.dilation(D, img_shape, struct_element)
                D = np.bitwise_and(dilated, remaining)
                if np.array_equal(D, D_prev):
                    break

            components.append((label, D))
            remaining = np.bitwise_and(remaining, ~D)
            label += 1

        return components

    @classmethod
    def convex_hull(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        struct_elements: List[StructElement],
    ) -> np.ndarray:
        result_matrix = img_matrix.copy()

        while True:
            result_prev = result_matrix.copy()

            for SE in struct_elements:
                D = result_matrix.copy()

                while True:
                    D_prev = D.copy()
                    hormiss = Morphology.hit_or_miss_simplified(D, img_shape, SE)
                    D = np.bitwise_or(hormiss, D)
                    if np.array_equal(D, D_prev):
                        break

                result_matrix = np.bitwise_or(result_matrix, D)

            if np.array_equal(result_matrix, result_prev):
                break

        return result_matrix

    @classmethod
    def thinning(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        struct_elements: List[StructElement],
    ) -> np.ndarray:
        result_matrix = img_matrix.copy()

        while True:
            result_prev = result_matrix.copy()

            for SE in struct_elements:
                hitormiss = Morphology.hit_or_miss_simplified(
                    result_matrix, img_shape, SE
                )
                result_matrix = np.bitwise_and(result_matrix, ~hitormiss)

            if np.array_equal(result_matrix, result_prev):
                break

        return result_matrix

    @classmethod
    def skeleton(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        struct_elements: List[StructElement],
    ) -> np.ndarray:
        result_matrix = img_matrix.copy()
        skel = np.zeros_like(img_matrix)

        while True:
            result_prev = result_matrix.copy()

            for SE in struct_elements:
                hitormiss = Morphology.hit_or_miss_simplified(
                    result_matrix, img_shape, SE
                )
                result_matrix = np.bitwise_and(result_matrix, ~hitormiss)

            diff = np.bitwise_and(result_prev, ~result_matrix)
            skel = np.bitwise_or(skel, diff)

            if np.array_equal(result_matrix, result_prev):
                break

        return skel

    @classmethod
    def pruning(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        struct_elements: List[StructElement],
        iterations: int = 3,
    ) -> np.ndarray:
        result_matrix = img_matrix.copy()

        for _ in range(iterations):
            for SE in struct_elements:
                hitormiss = Morphology.hit_or_miss_simplified(
                    result_matrix, img_shape, SE
                )
                result_matrix = np.bitwise_and(result_matrix, ~hitormiss)

        endpoints = np.zeros_like(img_matrix)
        for SE in struct_elements:
            hitormiss = Morphology.hit_or_miss_simplified(
                result_matrix, img_shape, SE
            )
            endpoints = np.bitwise_or(endpoints, hitormiss)

        for _ in range(iterations):
            endpoints = Morphology.dilation(endpoints, img_shape, struct_elements[0])
            endpoints = np.bitwise_and(endpoints, img_matrix)

        return np.bitwise_or(result_matrix, endpoints)
    # --- Operações grayscale (também funcionam para binária) ---

    @classmethod
    def morphological_gradient(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        kernel: StructElement,
    ) -> np.ndarray:
        dilated = Morphology.dilation(img_matrix, img_shape, kernel)
        eroded  = Morphology.erosion(img_matrix, img_shape, kernel)
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
