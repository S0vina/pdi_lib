from pdi import Image
import numpy as np


class IntensityTransforms:
    def __init__(self) -> None:
        pass

    @classmethod
    def threshold(cls, img: Image, thresshold: int = 127) -> np.ndarray:
        if not isinstance(img, Image):
            raise TypeError("The method needs an Image type object")

        thr_img = np.zeros_like(img.data)
        rows, colluns = img.shape

        for i in range(rows):
            for j in range(colluns):
                if img.data[i][j] > thresshold:
                    thr_img[i][j] = 255
                else:
                    thr_img[i][j] = 0

        return thr_img

    @classmethod
    def negative(cls, img: Image) -> np.ndarray:
        if not isinstance(img, Image):
            raise TypeError("The method needs an Image type object")

        neg_img = np.zeros_like(img.data)
        bigger_intesity = np.iinfo(img.data.dtype).max
        rows, colluns = img.shape

        for i in range(rows):
            for j in range(colluns):
                neg_img[i][j] = bigger_intesity - 1 - img.data[i][j]

        return neg_img

    @classmethod
    def logaritmic(cls, img: Image, const: int) -> np.ndarray:
        if not isinstance(img, Image):
            raise TypeError("The method needs an Image type object")

        log_img = np.zeros_like(img.data)
        rows, colums = img.shape

        for i in range(rows):
            for j in range(colums):
                log_img[i][j] = const * np.log(1 + img.data[i][j])

        return log_img
