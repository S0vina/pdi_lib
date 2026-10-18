import numpy as np


class Mask:
    MEAN = "mean"
    GAUSSIAN = "gaussian"

    @classmethod
    def null_shape(cls, height: int, width: int) -> np.ndarray:
        if height < 0 or width < 0:
            raise ValueError("height and width values should be greater than 0")

        mask = np.zeros((height, width), dtype=int)

        return mask

    @classmethod
    def from_type(cls, mask_type=None) -> np.ndarray:
        if mask_type is None:
            raise ValueError("mask type should be specified")

        if mask_type == cls.MEAN:
            pass

        if mask_type == cls.GAUSSIAN:
            pass
