from pdi.src.core.image import Image
import numpy as np


class Equalizer:
    def __init__(self):
        pass

    def cdf_eq_image(
        self, lut_hist: np.ndarray, src_img: Image, img_title: str
    ) -> Image:
        eq_array = np.zeros_like(src_img.data)
        rows, colluns = src_img.shape

        for i in range(rows):
            for j in range(colluns):
                eq_array[i][j] = lut_hist[src_img.data[i][j]]

        eq_img = Image(eq_array, img_title)
        return eq_img
