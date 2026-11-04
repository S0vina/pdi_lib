from .mask import Mask
from typing import List, Self, Tuple
import numpy as np


class StructElement(Mask):
    # Variaveis globais vão ser definidas dessa forma
    AFINAMENTO = "AFINAMENTO"
    PREENCHIMENTO_DE_BURACOS = "PREENCHIMENTO_DE_BURACOS"

    def __init__(self, array, mt, mn, h: int, w: int) -> None:
        super().__init__(array, mt, mn, h, w)

    @classmethod
    def any_shape(
        cls,
        mask_type,
        mask_neighboor=None,
        height: int = -1,
        width: int = -1,
        weight: float | bool = True,
    ) -> Self:

        if height < 0 or width < 0:
            raise ValueError("height and width values should be greater than 0")

        bool_weight = bool(weight)
        array = np.full((height, width), bool_weight, dtype=bool)

        return cls(array, mask_type, mask_neighboor, height, width)
