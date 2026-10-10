from .Image import Image
import numpy as np


class Filter:
    def __init__(self) -> None:
        pass

    @classmethod
    def bin_filter(cls, img: Image, thresshold: int = 127) -> np.ndarray:
        if not isinstance(img, Image):
            raise TypeError("The method needs an object of the type Image")

        bin_img = np.zeros_like(img.data)
        rows, colluns = img.shape

        for i in range(rows):
            for j in range(colluns):
                if img.data[i][j] > thresshold:
                    bin_img[i][j] = 255
                else:
                    bin_img[i][j] = 0

        return bin_img
