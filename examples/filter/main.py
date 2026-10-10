from pathlib import Path
from core.plugins.Filter import Filter
from core.plugins.Image import Image

if __name__ == "__main__":
    path = Path("../../assets/niki_lauda.jpg")

    data = Image.get_data_from_file(path)
    img = Image(data, "niki")

    print(img.data)
    print(img.shape)

    bin_data = Filter.bin_filter(img)
    bin_img = Image(bin_data, "bin_niki")

    img.plot_img()
    bin_img.plot_img()
