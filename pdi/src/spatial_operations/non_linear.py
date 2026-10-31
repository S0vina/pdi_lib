from pdi.src.core.image import Image
from pdi.src.core.mask import Mask
import numpy as np


class NonLinearSpatialFilter:
    def __init__(self) -> None:
        pass

    @classmethod
    def median(cls, img: Image, n: int = 3) -> np.ndarray:
        if n % 2 != 1:
            raise ValueError("Median method should have a quadratic roi")
        img_h, img_w = img.shape
        matriz = np.zeros((img_h, img_w), dtype=np.uint8)
        fit_mask = Mask.any_shape(None, None, n, n, 0)
        center = (n * n) // 2

        for i in range(img_h):
            for j in range(img_w):
                if fit_mask.it_fits(i, j, img.shape):
                    roi = fit_mask.get_list(i, j, img.data)
                    roi.sort()
                    matriz[i][j] = roi[center]

        return matriz

    @classmethod
    def max_filter(cls, img: Image, n: int = 3) -> np.ndarray:
        if n % 2 != 1:
            raise ValueError("Max method should have a qaudratic roi")

        img_h, img_w = img.shape
        matriz = np.zeros((img_h, img_w), dtype=np.uint8)
        fit_mask = Mask.any_shape(None, None, n, n, 1)
        pointer_value = (n * n) - 1

        for i in range(img_h):
            for j in range(img_w):
                if fit_mask.it_fits(i, j, img.shape):
                    roi = fit_mask.get_list(i, j, img.data)
                    roi.sort()
                    matriz[i][j] = roi[pointer_value]

        return matriz

    @classmethod
    def min_filter(cls, img: Image, n: int = 3) -> np.ndarray:
        if n % 2 != 1:
            raise ValueError("Max method should have a qaudratic roi")

        img_h, img_w = img.shape
        matriz = np.zeros((img_h, img_w), dtype=np.uint8)
        fit_mask = Mask.any_shape(None, None, n, n, 1)
        pointer_value = 0

        for i in range(img_h):
            for j in range(img_w):
                if fit_mask.it_fits(i, j, img.shape):
                    roi = fit_mask.get_list(i, j, img.data)
                    roi.sort()
                    matriz[i][j] = roi[pointer_value]

        return matriz

    @classmethod
    def adaptive_denoise(cls, img: Image, n: int, noise_variance: float) -> np.ndarray:
        if n % 2 == 0:
            raise ValueError("Window size (n) should'nt be a pair value")

        if noise_variance == 0.0:
            return img.data

        img_h, img_w = img.shape
        f_matrix = img.data.copy()
        sq_matrix = img.data.astype(np.float64) ** 2
        lp_mask = Mask.from_type(Mask.LOW_PASS, Mask.EIGHT_NEIGHBOR, n)

        for i in range(img_h):
            for j in range(img_w):
                if lp_mask.it_fits(i, j, img.shape):
                    local_avg = lp_mask.dot_product(
                        i, j, img.data, img.shape, float_type=True
                    )

                    local_variance = lp_mask.get_local_variance(
                        i, j, sq_matrix, img.shape, local_avg
                    )

                    if local_variance == 0:
                        p_intensity = local_avg

                    else:
                        p_intensity = img.data[i][j] - (
                            (noise_variance / local_variance)
                            * (img.data[i][j] - local_avg)
                        )
                        f_matrix[i][j] = int(p_intensity)

        return f_matrix
