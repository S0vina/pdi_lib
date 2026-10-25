from src.plugins.Image import Image
from src.plugins.NonLinearSpatialFilter import NonLinearSpatialFilter

if __name__ == "__main__":
    img_data = Image.get_data_from_file("../../assets/niki_lauda.jpg")
    img = Image(img_data, "niki")
    img.plot_img()

    median_data = NonLinearSpatialFilter.median(img)
    median_img = Image(median_data, "median_niki")
    median_img.plot_img()
