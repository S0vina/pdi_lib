from pathlib import Path
from pdi import IntensityTransforms
from pdi import Image

if __name__ == "__main__":
    path = Path("../../assets/niki_lauda.jpg")

    data = Image.get_data_from_file(path)
    img = Image(data, "niki")

    bin_data = IntensityTransforms.threshold(img)
    bin_img = Image(bin_data, "bin_niki")

    img.plot_img()
    bin_img.plot_img()

    neg_data = IntensityTransforms.negative(img)
    neg_img = Image(neg_data, "neg_niki")

    neg_img.plot_img()

    log_data = IntensityTransforms.logaritmic(img, 25)
    log_img = Image(log_data, "log_niki")
    log_img.plot_img()

    median_img.plot_img()
