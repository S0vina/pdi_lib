from src.plugins.Image import Image
from src.plugins.Mask import Mask
from src.plugins.LinearSpatialFilter import LinearSpatialFilter

if __name__ == "__main__":
    img_data = Image.get_data_from_file("../../assets/niki_lauda.jpg")
    img = Image(img_data, "niki_lauda")
    img.plot_img()
    print("imagem original")
    mask = Mask.any_shape(3, 3, 1)
    print("mascara produzida")
    # filter_data = LinearSpatialFilter.convolution(img, mask)
    # print("Convulução realizada")
    # filter_img = Image(filter_data, "filter_niki")
    # filter_img.plot_img()

    low_pass_data = LinearSpatialFilter.low_pass(img)
    print("filtro low pass aplicado")
    lp_img = Image(low_pass_data, "lp_niki")
    lp_img.plot_img()
    print("Tudo certo")
