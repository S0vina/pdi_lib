from pdi import Image
from pdi import StructElement
from pdi import IntensityTransforms
from pdi import Morphology

if __name__ == "__main__":
    img_data = Image.get_data_from_file("../../assets/cavalo_correndo_feliz.png")
    img = Image(img_data, "lena")

    lena_pb = IntensityTransforms.threshold(img, 240)
    lena_pb_img = Image(lena_pb, "lena_pb")

    lena_pb_img.plot_img()

    se = StructElement.any_shape(None, None, 17, 17)

    lena_dilatation = Morphology.erosion(lena_pb_img, se)
    l_dila = Image(lena_dilatation, "lena_dil")

    l_dila.plot_img()

    lena_abrida = Morphology.opening(lena_pb_img, se)
    lena_abrida_img = Image(lena_abrida, "lena_open")

    lena_abrida_img.plot_img()

    lena_fechada = Morphology.closing(lena_abrida_img, se)
    lena_fechadinha = Image(lena_fechada, "lena_closes")

    lena_fechadinha.plot_img()
