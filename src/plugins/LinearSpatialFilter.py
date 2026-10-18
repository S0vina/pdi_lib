from src.plugins.Image import Image
from src.plugins.Mask impor Mask
import numpy as np

class LinearSpatialFilter():
    def __init__(self) -> None:
        pass
    
    @classmethod
    def correlation(cls, img: Image, f: Mask) -> np.ndarray:
        pass
    
    @classmethod
    def convolution(cls, img: Image, f: Mask) -> np.ndarray:
        pass



