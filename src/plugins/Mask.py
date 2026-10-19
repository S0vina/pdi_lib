from typing import Self
import numpy as np


class Mask:
    LOW_PASS = "low_pass"
    GAUSSIAN = "gaussian"

    def __init__(self):
        self.height = 0
        self.width = 0
        self.array = None

    @property
    def shape(self):
        return (self.height, self.width)

    @classmethod
    def null_shape(cls, height: int, width: int) -> Self:
        if height < 0 or width < 0:
            raise ValueError("height and width values should be greater than 0")

        mask = cls()
        mask.array = np.zeros((height, width), dtype=int)
        mask.height = height
        mask.width = width

        return mask

    @classmethod
    def from_type(cls, mask_type=None, n: int = 3) -> Self:
        if mask_type is None:
            raise ValueError("mask type should be specified")

        if n % 2 != 1:
            raise ValueError("a pair mask is not implemented in this lib")

        mask = cls()
        mask.height = n
        mask.width = n

        if mask_type == cls.LOW_PASS:
            mask.array = np.full((n, n), 1, dtype=int)

        if mask_type == cls.GAUSSIAN:
            pass

        return mask
