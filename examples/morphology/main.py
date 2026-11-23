from matplotlib.pyplot import fill, plot

from pdi import Image
from pdi import StructElement
from pdi import IntensityTransforms
from pdi import BinaryMorphology
from pdi import MorphologicalOps

if __name__ == "__main__":
    src_img = Image.get_data("../../assets/buracos.png")

    bw_img = IntensityTransforms.threshold(src_img, src_img.shape)

    print("=== Source Image ===")
    Image.plot(bw_img)

    kernel = StructElement.any_shape(None, None, 3, 3)

    dilatation_img = BinaryMorphology.dilatation(bw_img, bw_img.shape, kernel)

    print("=== Dilated Image ===")
    Image.plot(dilatation_img)

    erosion_img = BinaryMorphology.erosion(bw_img, bw_img.shape, kernel)

    print("=== erosão ===")
    Image.compare(bw_img, erosion_img)

    # open_img = MorphologicalOps.opening(bw_img, bw_img.shape, kernel)

    # print("=== opening ===")
    # Image.plot(open_img)

    # close_img = MorphologicalOps.closing(bw_img, bw_img.shape, kernel)

    # print("=== Closing ===")
    # Image.plot(close_img)

    boundary_img = MorphologicalOps.boudary_extraction(bw_img, bw_img.shape, kernel)

    print("=== Boundary Extraction ===")
    Image.plot(boundary_img)

    # kernels = [#list of kernels for convex_hull]
    # convex_img = MorphologicalOps.convex_hull(bw_img, bw_img.shape, kernels)

    # print("=== Convex Hull ===")
    # Image.plot(convex_img)

    filling_img = MorphologicalOps.hole_filling(bw_img, bw_img.shape, kernel)

    print("=== Hole Filling ===")
    Image.plot(filling_img)

    # components = MorphologicalOps.connected_components_extraction(
    #    bw_img, bw_img.shape, kernel
    # )

    # print("=== Connected Components ===")
    # print(components)
