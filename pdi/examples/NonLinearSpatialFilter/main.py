from src.plugins.Image import Image
from src.plugins.NonLinearSpatialFilter import NonLinearSpatialFilter

if __name__ == "__main__":
    img_data = Image.get_data_from_file("../../assets/niki_lauda.jpg")
    img = Image(img_data, "niki")
    print("Imagem original carregada")
    img.plot_img()

    # --- MEDIAN FILTER ---
    median_data = NonLinearSpatialFilter.median(img)
    # Data do filtro aplicado capturada
    median_img = Image(median_data, "median_niki")
    # plotando imagem filtrada
    median_img.plot_img()

    # --- MAX  FILTER ---
    max_data = NonLinearSpatialFilter.max_filter(img)
    max_img = Image(max_data, "max_niki")
    max_img.plot_img()

    # --- MIN  FILTER ---
    min_data = NonLinearSpatialFilter.min_filter(img)
    min_img = Image(min_data, "min_niki")
    min_img.plot_img()
