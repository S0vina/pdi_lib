from pdi.src.core.image import Image
from pdi.src.core.morphology.struct_element import StructElement
from pdi.src.spatial_operations.morphological_ops import BinaryMorphology
import numpy as np


class MorphologicalReconstructors:
    def __init__(self) -> None:
        pass

    @classmethod
    def reconstruction_by_dilation(
        cls, marker: Image, mask: Image, struct_element: StructElement
    ):
        result_matrix = marker.data.copy()

        while True:
            result_prev = result_matrix.copy()

            dilated = BinaryMorphology.dilatation(
                result_matrix, marker.shape, struct_element
            )
            result_matrix = np.logical_and(dilated, mask.data).astype(np.uint8)

            if np.array_equal(result_matrix, result_prev):
                break

        return result_matrix

    @classmethod
    def reconstruction_by_erosion(
        cls, marker: Image, mask: Image, struct_element: StructElement
    ):
        result_matrix = marker.data.copy()

        while True:
            result_prev = result_matrix.copy()

            eroded = BinaryMorphology.erosion(
                result_matrix, marker.shape, struct_element
            )
            result_matrix = np.logical_or(eroded, mask.data).astype(np.uint8)

            if np.array_equal(result_matrix, result_prev):
                break

        return result_matrix

    @classmethod
    def opening_by_reconstruction(cls, img: Image, struct_element: StructElement):
        # Step 1: erode to get marker
        marker_data = BinaryMorphology.erosion(img.data, img.shape, struct_element)
        marker = Image(marker_data, "any")  # change this later

        # Step 2: reconstruct by dilation back to mask
        return cls.reconstruction_by_dilation(marker, img, struct_element)

    @classmethod
    def closing_by_reconstruction(cls, img: Image, struct_element: StructElement):
        # Step 1: dilate to get marker
        marker_data = BinaryMorphology.dilatation(img.data, img.shape, struct_element)
        marker = Image(marker_data, "any")

        # Step 2: reconstruct by erosion back to mask
        return cls.reconstruction_by_erosion(marker, img, struct_element)

    @classmethod
    def clear_border_objects(cls, img: Image, struct_element: StructElement):
        # Step 1: create marker from border pixels only
        marker_data = np.zeros_like(img.data)
        marker_data[0, :] = img.data[0, :]  # top border
        marker_data[-1, :] = img.data[-1, :]  # bottom border
        marker_data[:, 0] = img.data[:, 0]  # left border
        marker_data[:, -1] = img.data[:, -1]  # right border

        marker = Image(marker_data, "any")  # fix that later

        # Step 2: reconstruct by dilation — grows marker through
        # every object connected to the border
        border_objects = cls.reconstruction_by_dilation(marker, img, struct_element)

        # Step 3: subtract border objects from original
        result = np.logical_and(img.data, ~border_objects).astype(np.uint8)

        return result
