import numpy as np
from .struct_element import StructElement


class GrayScaleMorphology(StructElement):
    def __init__(self, array, mt, mn, h: int, w: int) -> None:
        super().__init__(array, mt, mn, h, w)
