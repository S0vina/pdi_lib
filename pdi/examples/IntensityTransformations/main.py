from pathlib import Path
from src.plugins.IntesityTransformations import IntesityTransforms
from src.plugins.Image import Image
from src.plugins.NonLinearSpatialFilter import NonLinearSpatialFilter

if __name__ == "__main__":
    path = Path("../../assets/niki_lauda.jpg")

    data = Image.get_data_from_file(path)
    img = Image(data, "niki")

    bin_data = IntesityTransforms.threshold(img)
    bin_img = Image(bin_data, "bin_niki")

    img.plot_img()
    bin_img.plot_img()

    neg_data = IntesityTransforms.negative(img)
    neg_img = Image(neg_data, "neg_niki")

    neg_img.plot_img()

    log_data = IntesityTransforms.logaritmic(img, 25)
    log_img = Image(log_data, "log_niki")
    log_img.plot_img()

    median_data = NonLinearSpatialFilter.median(neg_img)
    median_img = Image(neg_data, "neg_niki")

    median_img.plot_img()
