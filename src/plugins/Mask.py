from typing import List, Self, Tuple
import numpy as np


class Mask:
    LOW_PASS = "low_pass"
    GAUSSIAN = "gaussian"

    def __init__(self, array, h: int, w: int):
        self.array = array
        self.height = h
        self.width = w

    @property
    def shape(self):
        return (self.height, self.width)

    @classmethod
    def any_shape(cls, height: int, width: int, weight: float) -> Self:
        if height < 0 or width < 0:
            raise ValueError("height and width values should be greater than 0")

        array = np.full((height, width), weight)

        return cls(array, height, width)

    @classmethod
    def from_type(cls, mask_type, n: int = 3) -> Self:
        print(mask_type)
        if mask_type is None:
            raise ValueError("mask type should be specified")

        if n % 2 != 1:
            raise ValueError("a pair mask is not implemented in this lib")

        if mask_type == cls.LOW_PASS:
            weight = 1.0 / (n * n)
            return cls.any_shape(n, n, weight)

        if mask_type == cls.GAUSSIAN:
            pass

        raise ValueError(f"Unknown mask type: {mask_type}")

    def it_fits(self, i: int, j: int, img_t: Tuple[int, int]) -> bool:
        img_h, img_w = img_t
        mask_h, mask_w = self.shape

        # Verifica se a máscara ultrapassa a altura ou largura da imagem
        if i + mask_h > img_h:
            return False
        if j + mask_w > img_w:
            return False

        return True

    def get_list(self, center_i: int, center_j: int, img_data: np.ndarray) -> list:
        mask_h, mask_w = self.shape
        flat_roi = []

        half_h = mask_h // 2
        half_w = mask_w // 2

        for r in range(center_i - half_h, center_i + half_h + 1):
            for c in range(center_j - half_w, center_j + half_w + 1):
                flat_roi.append(img_data[r][c])

        return flat_roi

    @classmethod
    def flip(cls, mask: Self) -> Self:
        mask_h, mask_w = mask.shape
        array = np.zeros((mask_h, mask_w))
        for i in range(mask_h):
            for j in range(mask_w):
                array[i][j] = mask.array[mask_h - i - 1][mask_w - j - 1]

        return cls(array, mask_h, mask_w)
