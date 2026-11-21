from pdi import Image
from pdi import StructElement
from pdi import IntensityTransforms
from pdi import BinaryMorphology
from pdi import MorphologicalOps

if __name__ == "__main__":
    img_data = Image.get_data_from_file("../../assets/rage_gray.png")
    img = Image(img_data, "cavalos")

    cavalo_pb = IntensityTransforms.threshold(img, 240)
    cavalo_pb = Image(cavalo_pb, "")

    cavalo_pb.plot_img()

    kernel = StructElement.any_shape(None, None, 7, 7)

    cavalo_erosion = BinaryMorphology.erosion(cavalo_pb.data, cavalo_pb.shape, kernel)
    cavalo_erosion = Image(cavalo_erosion, "lena_dil")

    cavalo_pb.plot_img()

    print("=== erosão ===")

    cavalo_open = MorphologicalOps.opening(cavalo_pb.data, cavalo_pb.shape, kernel)
    cavalo_open = Image(cavalo_open, "lena_open")

    cavalo_open.plot_img()

    print("=== opening ===")

    cavalo_close = MorphologicalOps.closing(cavalo_pb.data, cavalo_pb.shape, kernel)
    cavalo_close = Image(cavalo_close, "lena_closes")

    cavalo_close.plot_img()

    print("=== Closing ===")
