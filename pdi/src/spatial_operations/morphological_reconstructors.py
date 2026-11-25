from pdi.src.core.image import Image
from pdi.src.core.morphology.struct_element import StructElement
from pdi.src.core.morphology.morphology import Morphology
import numpy as np


class MorphologicalReconstructors:
    """
    Reconstrução morfológica por dilatação e erosão iterativas.
    Opera em imagens binárias {0, 255}. Usa bitwise_and/or para manter
    a convenção objeto=255 durante todo o loop iterativo.
    """

    @classmethod
    def reconstruction_by_dilation(
        cls,
        marker: Image,
        mask: Image,
        struct_element: StructElement,
    ) -> np.ndarray:
        """
        Reconstrói pelo marcador crescendo via dilatação até ser contido pela máscara.
        marker ⊆ mask (ambas {0,255}).
        """
        result_matrix = marker.data.copy()

        while True:
            result_prev = result_matrix.copy()
            dilated = Morphology.dilation(result_matrix, marker.shape, struct_element)
            # interseção em {0,255}
            result_matrix = np.bitwise_and(dilated, mask.data)
            if np.array_equal(result_matrix, result_prev):
                break

        return result_matrix

    @classmethod
    def reconstruction_by_erosion(
        cls,
        marker: Image,
        mask: Image,
        struct_element: StructElement,
    ) -> np.ndarray:
        """
        Reconstrói pelo marcador encolhendo via erosão até conter a máscara.
        marker ⊇ mask (ambas {0,255}).
        """
        result_matrix = marker.data.copy()

        while True:
            result_prev = result_matrix.copy()
            eroded = Morphology.erosion(result_matrix, marker.shape, struct_element)
            # união em {0,255}
            result_matrix = np.bitwise_or(eroded, mask.data)
            if np.array_equal(result_matrix, result_prev):
                break

        return result_matrix

    @classmethod
    def opening_by_reconstruction(
        cls,
        img: Image,
        struct_element: StructElement,
    ) -> np.ndarray:
        marker_data = Morphology.erosion(img.data, img.shape, struct_element)
        marker = Image(marker_data, "any")
        return cls.reconstruction_by_dilation(marker, img, struct_element)

    @classmethod
    def closing_by_reconstruction(
        cls,
        img: Image,
        struct_element: StructElement,
    ) -> np.ndarray:
        marker_data = Morphology.dilation(img.data, img.shape, struct_element)
        marker = Image(marker_data, "any")
        return cls.reconstruction_by_erosion(marker, img, struct_element)

    @classmethod
    def clear_border_objects(
        cls,
        img: Image,
        struct_element: StructElement,
    ) -> np.ndarray:
        marker_data = np.zeros_like(img.data)
        marker_data[0, :]  = img.data[0, :]
        marker_data[-1, :] = img.data[-1, :]
        marker_data[:, 0]  = img.data[:, 0]
        marker_data[:, -1] = img.data[:, -1]

        marker = Image(marker_data, "any")
        border_objects = cls.reconstruction_by_dilation(marker, img, struct_element)

        # remove objetos de borda: interseção com complemento
        return np.bitwise_and(img.data, ~border_objects)
