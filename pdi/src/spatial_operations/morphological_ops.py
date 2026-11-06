from pdi.src.core.morphology.struct_element import StructElement
from pdi.src.core.morphology.binary import BinaryMorphology
from pdi.src.core.image import Image
from typing import List
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

        x_matrix = np.zeros_like(img.data)

        x_matrix[0, :] = not_img[0, :]
        x_matrix[-1, :] = not_img[-1, :]
        x_matrix[:, 0] = not_img[:, 0]
        x_matrix[:, -1] = not_img[:, -1]

        while True:
            k_minusone_matrix = np.copy(x_matrix)
            dilated = BinaryMorphology.dilatation(
                x_matrix, img.shape, struct_element
            ).astype(np.uint8)
            x_matrix = np.logical_and(dilated, not_img).astype(np.uint8)
            # no change between dilatations
            if np.array_equal(k_minusone_matrix, x_matrix):
                break

        holes = not_img - x_matrix

        hole_filed = np.logical_or(holes, img.data)
        return hole_filed

    @classmethod
    def connected_components_extraction(cls, img: Image, struct_element: StructElement):
        remaining = img.data.copy()
        components = []
        label = 1

        remaining_h, remaining_w = img.shape

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

            D = np.zeros_like(img.data)
            D[seed] = 1  # plant the seed

            while True:
                D_prev = D.copy()
                dilated = BinaryMorphology.dilatation(D, img.shape, struct_element)
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
    def convex_hull(cls, img: Image, struct_elements: List[StructElement]):
        result_matrix = img.data.copy()

        while True:
            result_prev = result_matrix.copy()

            for SE in struct_elements:
                D = result_matrix.copy()

                while True:
                    D_prev = D.copy()
                    hormiss = BinaryMorphology.hit_or_miss_simplified(D, img.shape, SE)
                    D = np.logical_or(hormiss, D).astype(np.uint8)
                    if np.array_equal(D, D_prev):
                        break

                result_matrix = np.logical_or(result_matrix, D).astype(np.uint8)

            if np.array_equal(result_matrix, result_prev):
                break

        return result_matrix


@classmethod
def tining(cls, img: Image, struct_elements: List[StructElement]):
    result_matrix = img.data.copy()

    while True:
        result_prev = result_matrix.copy()

        for SE in struct_elements:
            hitormiss = BinaryMorphology.hit_or_miss_simplified(
                result_matrix, img.shape, SE
            )
            result_matrix = np.logical_and(result_matrix, ~hitormiss).astype(np.uint8)

        if np.array_equal(result_matrix, result_prev):
            break

    return result_matrix


@classmethod
def skeleton(cls, img: Image, struct_elements: List[StructElement]):
    result_matrix = img.data.copy()
    skeleton = np.zeros_like(img.data)

    while True:
        result_prev = result_matrix.copy()

        for SE in struct_elements:
            hitormiss = BinaryMorphology.hit_or_miss_simplified(
                result_matrix, img.shape, SE
            )
            result_matrix = np.logical_and(result_matrix, ~hitormiss).astype(np.uint8)

        # accumulate removed pixels into skeleton
        skeleton = np.logical_or(skeleton, result_prev - result_matrix).astype(np.uint8)

        if np.array_equal(result_matrix, result_prev):
            break

    return skeleton


@classmethod
def pruning(cls, img: Image, struct_elements: List[StructElement], iterations: int = 3):
    result_matrix = img.data.copy()

    # Step 1: remove parasitic branches with thinning-like passes
    for _ in range(iterations):
        for SE in struct_elements:
            hitormiss = BinaryMorphology.hit_or_miss_simplified(
                result_matrix, img.shape, SE
            )
            result_matrix = np.logical_and(result_matrix, ~hitormiss).astype(np.uint8)

    # Step 2: recover endpoints that were removed
    endpoints = np.zeros_like(img.data)
    for SE in struct_elements:
        hitormiss = BinaryMorphology.hit_or_miss_simplified(
            result_matrix, img.shape, SE
        )
        endpoints = np.logical_or(endpoints, hitormiss).astype(np.uint8)

    # Step 3: dilate endpoints and intersect with original to reconnect
    for _ in range(iterations):
        endpoints = BinaryMorphology.dilatation(
            endpoints, img.shape, struct_elements[0]
        )
        endpoints = np.logical_and(endpoints, img.data).astype(np.uint8)

    result_matrix = np.logical_or(result_matrix, endpoints).astype(np.uint8)

    return result_matrix
