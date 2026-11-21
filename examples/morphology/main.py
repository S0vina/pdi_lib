from pdi import Image
from pdi import StructElement
from pdi import IntensityTransforms
from pdi import BinaryMorphology
from pdi import MorphologicalOps

if __name__ == "__main__":
    src_img = Image.get_data("../../assets/rage_gray.png")

    bw_img = IntensityTransforms.threshold(src_img, src_img.shape)

    print("=== Source Image ===")
    Image.plot(bw_img)

    kernel = StructElement.any_shape(None, None, 7, 7)

    erosion_img = BinaryMorphology.erosion(bw_img, bw_img.shape, kernel)

    print("=== erosão ===")
    Image.plot(erosion_img)

    open_img = MorphologicalOps.opening(bw_img, bw_img.shape, kernel)

    print("=== opening ===")
    Image.plot(open_img)

    close_img = MorphologicalOps.closing(bw_img, bw_img.shape, kernel)

    print("=== Closing ===")
    Image.plot(close_img)
