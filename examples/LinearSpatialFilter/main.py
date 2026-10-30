from pdi import Image
from pdi import LinearSpatialFilter

if __name__ == "__main__":
    img_data = Image.get_data_from_file("../../assets/niki_lauda.jpg")
    img = Image(img_data, "niki_lauda")
    img.plot_img()
    print("imagem original")

    low_pass_data = LinearSpatialFilter.low_pass(img)
    print("filtro low pass aplicado")
    lp_img = Image(low_pass_data, "lp_niki")
    lp_img.plot_img()

    laplassian_data = LinearSpatialFilter.laplassian(img)
    laplassian_img = Image(laplassian_data, "la_niki")
    laplassian_img.plot_img()
