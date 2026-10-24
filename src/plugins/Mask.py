from typing import Self, Tuple
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
    def null_shape(cls, height: int, width: int) -> Self:
        if height < 0 or width < 0:
            raise ValueError("height and width values should be greater than 0")

        array = np.zeros((height, width))

        return cls(array, height, width)

    @classmethod
    def from_type(cls, mask_type=None, n: int = 3) -> Self:
        if mask_type is None:
            raise ValueError("mask type should be specified")

        if n % 2 != 1:
            raise ValueError("a pair mask is not implemented in this lib")

        if mask_type == cls.LOW_PASS:
            array = np.full((n, n), 1, dtype=int)
            return cls(array, n, n)

        if mask_type == cls.GAUSSIAN:
            pass

        return cls(-1, -1, -1)

    def it_fits(self, i: int, j: int, img_t: Tuple[int, int]) -> bool:
        img_h, img_w = img_t
        mask_h, mask_w = self.shape

        # Verifica se a máscara ultrapassa a altura ou largura da imagem
        if i + mask_h > img_h:
            return False
        if j + mask_w > img_w:
            return False

        return True

    @classmethod
    def flip(cls, m: Self) -> Self:
        mask_h, mask_w = m.shape
        instance = cls.null_shape(mask_h, mask_w)

        for i in range(mask_h):
            for j in range(mask_w):
                instance.array[i][j] = m.array[mask_h - i - 1][mask_w - j - 1]

        return instance
