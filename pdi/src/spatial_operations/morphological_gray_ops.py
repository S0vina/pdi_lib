from re import I
from warnings import resetwarnings

from pdi.src.core.image import Image
from pdi.src.core.morphology.grayscale import GrayScaleMorphology
from pdi.src.core.morphology.grayscale import StructElement


class MorphologicalGrayOps:
    def __init__(self) -> None:
        pass

    @classmethod
    def opening(cls, img: Image, kernel: StructElement):
        erosion = GrayScaleMorphology.erosion(img.data, img.shape, kernel)

        result = GrayScaleMorphology.dilatation(erosion, img.shape, kernel)

        return result

    @classmethod
    def closing(cls, img: Image, kernel: StructElement):
        dilatation = GrayScaleMorphology.dilatation(img.data, img.shape, kernel)

        result = GrayScaleMorphology.erosion(dilatation, img.shape, kernel)

        return result

    @classmethod
    def morphological_gradient(cls, img: Image, kernel: StructElement):
        dilatation = GrayScaleMorphology.dilatation(img.data, img.shape, kernel)
        erosion = GrayScaleMorphology.erosion(img.data, img.shape, kernel)

        result = dilatation - erosion

        return result

    @classmethod
    def top_hat_transform(cls, img: Image, kernel: StructElement):
        opening = cls.opening(img, kernel)
        result = img.data - opening

        return result

    @classmethod
    def bottom_hat_transform(cls, img: Image, kernel: StructElement):
        closing = cls.closing(img, kernel)
        result = img.data - closing

        return result
