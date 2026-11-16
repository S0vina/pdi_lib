from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from .mask import Mask
from typing import Tuple, Self
import numpy as np
import math


class Avaliate:
    def __init__(self) -> None:
        pass

    @classmethod
    def psnr(
        cls, src_img: np.ndarray, proc_img: np.ndarray, img_shape: Tuple[int, int]
    ):
        """
        args:
            ref_img: original image (uint8)
            proc_img: processed image (uint8)
            img_shape: image's dimentions
        """

        def mse(src_img: np.ndarray, p_img: np.ndarray, img_s: Tuple[int, int]):
            img_h, img_w = src_img
            weight = img_h * img_w
            sum = 0.0

            for i in range(img_h):
                for j in range(img_w):
                    sum += (r_img[i][j] - p_img[i][j]) ** 2

            mse = sum / weight

            return mse

        mse_val = mse(src, proc_img, img_shape)

        if mse_val == 0:
            return float("inf")

        max_val = np.iinfo(ref_img.dtype).max

        psnr_val = 20 * math.log10(max_val / math.sqrt(mse_val))

        return psnr_val

    @classmethod
    def ssim(cls, ref_img: )

    @classmethod
    def avaliate(cls, img_ref, img_proc, label=""):
        """
        args:
            img_ref  : im1agem original (uint8, cinza)
            img_proc : imagem após filtragem (uint8, cinza)

        return:
            p, s
        """
        p = psnr(img_ref, img_proc, data_range=255)

        is_color = img_ref.ndim == 3
        s = ssim(
            img_ref, img_proc, data_range=255, channel_axis=-1 if is_color else None
        )

        return p, s
