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

    @classmethod
    def adaptive_median(cls, img: Image, n: int = 3, max_mask: int = 11) -> np.ndarray:
        if n % 2 == 0:
            raise ValueError("Window size (n) should'nt be a pair value")

        img_h, img_w = img.shape
        result_img = np.zeros((img_h, img_w))

        for i in range(img_h):
            for j in range(img_w):
                current_n = n
                pixel_processed = False

                while not pixel_processed and current_n < max_mask:
                    mask = Mask.any_shape(
                        None, Mask.EIGHT_NEIGHBOR, current_n, current_n, 1
                    )
                    if not mask.it_fits(i, j, img.shape):
                        result_img[i][j] = img.data[i][j]
                        break

                    flat_roi = mask.get_list(i, j, img.data)
                    flat_roi.sort()

                    # --- STAGE A ---
                    z_min = flat_roi[0]
                    z_med = flat_roi[len(flat_roi) // 2]
                    z_max = flat_roi[-1]

                    # verifica se não é ruido
                    if z_min < z_med < z_max:
                        # --- STAGE B ---
                        z_xy = img.data[i][j]

                        if z_min < z_xy < z_max:
                            result_img[i][j] = z_xy

                        else:
                            result_img[i][j] = z_med

                        pixel_processed = True

                    else:
                        current_n += 2

                if not pixel_processed:
                    result_img[i][j] = z_med

        return result_img
