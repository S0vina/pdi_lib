import numpy as np


class Mask:
    LOW_PASS = "low_pass"
    GAUSSIAN = "gaussian"

    @classmethod
    def null_shape(cls, height: int, width: int) -> np.ndarray:
        if height < 0 or width < 0:
            raise ValueError("height and width values should be greater than 0")

        mask = np.zeros((height, width), dtype=int)

        return mask

    @classmethod
    def from_type(cls, mask_type=None, n: int = 3) -> np.ndarray:
        if mask_type is None:
            raise ValueError("mask type should be specified")

        if n % 2 != 1:
            raise ValueError("a pair mask is not implemented in this lib")

        if mask_type == cls.LOW_PASS:
            lw_mask = np.full((n, n), 1, dtype=int)

            return lw_mask

        if mask_type == cls.GAUSSIAN:
            pass

        return np.full((n, n), -1)
