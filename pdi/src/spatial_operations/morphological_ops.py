from pdi.src.core.morphology.struct_element import StructElement
from pdi.src.core.morphology.binary import BinaryMorphology
from typing import List, Tuple
import numpy as np


class MorphologicalOps:
    def __init__(self) -> None:
        pass

    @classmethod
    def opening(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        struct_element: StructElement,
    ) -> np.ndarray:
        erosion_matrix = BinaryMorphology.erosion(img_matrix, img_shape, struct_element)

        dilation_matrix = BinaryMorphology.dilatation(
            erosion_matrix, img_shape, struct_element
        )

        return dilation_matrix

    @classmethod
    def closing(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        struct_element: StructElement,
    ) -> np.ndarray:
        dilation_matrix = BinaryMorphology.dilatation(
            img_matrix, img_shape, struct_element
        )

        erosion_matrix = BinaryMorphology.erosion(
            dilation_matrix, img_shape, struct_element
        )

        return erosion_matrix

    @classmethod
    def boudary_extraction(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        struct_element: StructElement,
    ) -> np.ndarray:
        erosion_matrix = BinaryMorphology.erosion(img_matrix, img_shape, struct_element)

        boundary_matrix = img_matrix - erosion_matrix

        return boundary_matrix

    @classmethod
    def hole_filling(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        struct_element: StructElement,
    ) -> np.ndarray:
        not_img = 255 - img_matrix

        x_matrix = np.zeros_like(img_matrix)

        x_matrix[0, :] = not_img[0, :]
        x_matrix[-1, :] = not_img[-1, :]
        x_matrix[:, 0] = not_img[:, 0]
        x_matrix[:, -1] = not_img[:, -1]

        while True:
            k_minusone_matrix = np.copy(x_matrix)
            dilated = BinaryMorphology.dilatation(
                x_matrix, img_shape, struct_element
            ).astype(np.uint8)
            x_matrix = np.bitwise_and(dilated, not_img).astype(np.uint8)
            # no change between dilatations
            if np.array_equal(k_minusone_matrix, x_matrix):
                break

        holes = not_img - x_matrix

        hole_filed = np.bitwise_or(holes, img_matrix)
        return hole_filed

    @classmethod
    def connected_components_extraction(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        struct_element: StructElement,
    ):
        remaining = img_matrix.copy()
        components = []
        label = 1

        remaining_h, remaining_w = img_shape

        while True:
            seed = None

            for i in range(remaining_h):
                for j in range(remaining_w):
                    if remaining[i][j] == 0:
                        seed = (i, j)

                    if seed:
                        break

                if seed:
                    break

            if seed is None:
                break

            D = np.zeros_like(img_matrix)
            D[seed] = 1  # plant the seed

            while True:
                D_prev = D.copy()
                dilated = BinaryMorphology.dilatation(D, img_shape, struct_element)
                D = np.logical_and(dilated, remaining).astype(np.uint8)
                if np.array_equal(D, D_prev):
                    break

            components.append((label, D))
            remaining = np.logical_and(remaining, ~D).astype(
                np.uint8
            )  # remove found component
            label += 1

        return components

    @classmethod
    def convex_hull(
        cls,
        img_matrix: np.ndarray,
        img_shape: Tuple[int, int],
        struct_elements: List[StructElement],
    ):
        result_matrix = img_matrix.copy()

        while True:
            result_prev = result_matrix.copy()

            for SE in struct_elements:
                D = result_matrix.copy()

                while True:
                    D_prev = D.copy()
                    hormiss = BinaryMorphology.hit_or_miss_simplified(D, img_shape, SE)
                    D = np.logical_or(hormiss, D).astype(np.uint8)
                    if np.array_equal(D, D_prev):
                        break

                result_matrix = np.logical_or(result_matrix, D).astype(np.uint8)

            if np.array_equal(result_matrix, result_prev):
                break

        return result_matrix


@classmethod
def tining(
    cls,
    img_matrix: np.ndarray,
    img_shape: Tuple[int, int],
    struct_elements: List[StructElement],
):
    result_matrix = img_matrix.copy()

    while True:
        result_prev = result_matrix.copy()

        for SE in struct_elements:
            hitormiss = BinaryMorphology.hit_or_miss_simplified(
                result_matrix, img_shape, SE
            )
            result_matrix = np.logical_and(result_matrix, ~hitormiss).astype(np.uint8)

        if np.array_equal(result_matrix, result_prev):
            break

    return result_matrix


@classmethod
def skeleton(
    cls,
    img_matrix: np.ndarray,
    img_shape: Tuple[int, int],
    struct_elements: List[StructElement],
):
    result_matrix = img_matrix.copy()
    skeleton = np.zeros_like(img_matrix)

    while True:
        result_prev = result_matrix.copy()

        for SE in struct_elements:
            hitormiss = BinaryMorphology.hit_or_miss_simplified(
                result_matrix, img_shape, SE
            )
            result_matrix = np.logical_and(result_matrix, ~hitormiss).astype(np.uint8)

        # acumula os pixels removidos do esqueleto
        skeleton = np.logical_or(skeleton, result_prev - result_matrix).astype(np.uint8)

        if np.array_equal(result_matrix, result_prev):
            break

    return skeleton


@classmethod
def pruning(
    cls,
    img_matrix: np.ndarray,
    img_shape: Tuple[int, int],
    struct_elements: List[StructElement],
    iterations: int = 3,
):
    result_matrix = img_matrix.copy()

    # Removendo partes parasitas com afinamento
    for _ in range(iterations):
        for SE in struct_elements:
            hitormiss = BinaryMorphology.hit_or_miss_simplified(
                result_matrix, img_shape, SE
            )
            result_matrix = np.logical_and(result_matrix, ~hitormiss).astype(np.uint8)

    # Recuperando "endpoints" que foram removidos
    endpoints = np.zeros_like(img_shape)
    for SE in struct_elements:
        hitormiss = BinaryMorphology.hit_or_miss_simplified(
            result_matrix, img_shape, SE
        )
        endpoints = np.logical_or(endpoints, hitormiss).astype(np.uint8)
    # Dilatando "endpoints" e realizando a intersecção com a imagem original para reconectar

    for _ in range(iterations):
        endpoints = BinaryMorphology.dilatation(
            endpoints, img_shape, struct_elements[0]
        )
        endpoints = np.logical_and(endpoints, img_matrix).astype(np.uint8)

    result_matrix = np.logical_or(result_matrix, endpoints).astype(np.uint8)

    return result_matrix
