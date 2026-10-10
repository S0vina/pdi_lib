from core.Image import Image
from core.Histogram import Histogram
from core.Equalizer import Equalizer
from pathlib import Path

if __name__ == "__main__":
    # --- PASSO 1 - Leitura e análise da imagem ---
    # Caminho da imagem original
    path = Path("../assets/polen.png")

    # Matriz da imagem original carregada na variável data
    data = Image.get_data_from_file(path)

    # criação de um objeto Image "standard image", para uso de funções da classe
    src_img = Image(data, "polen")

    # Exibição da imagem original
    src_img.plot_img()

    # Criação de um objeto da classe Histogram para uso de funções
    hist = Histogram("src_img Histogram")

    # Crição do histograma da imagem original
    src_hist = hist.img_histogram(src_img)
    # Exibição do histograma da imagem
    hist.plot_histogram(src_hist, "src_img Histogram")

    # --- PASSO 2 - Equalização do histograma
    # calcula a resolução a partir das medidas de altura e largura da imagem
    resolution = src_img.shape[0] * src_img.shape[1]

    # "look up table" a partir do histograma original
    src_lut = hist.lut(src_hist, resolution)

    # Exibição da "look up table"
    # hist.plot_histogram(src_lut, "lut")

    # Criação do histograma equalizado a partir do original e da "look up table"
    equalized_hist = hist.cdf_equalizer(src_hist, src_lut)

    # Criação de objeto da classe Equalizer para uso das funções
    eq = Equalizer()

    # Imagem equalizada a partir da função de equalização
    equalized_img = eq.cdf_eq_image(src_lut, src_img, "src_eq")

    # exibição das imagens original e equalizada
    src_img.plot_img()
    equalized_img.plot_img()
    hist.plot_comparison(
        src_hist,
        equalized_hist,
        "Source Histogram",
        "Equalized Histogram",
        "Histogram Comparation",
    )

    # --- PASSO 3 - Especificação do histograma ---
    # Carregando o caminho da imagem de referência
    esp_path = Path("../assets/niki_lauda.jpg")

    # adquirindo a matriz da imagem de referência
    esp_data = Image.get_data_from_file(esp_path)

    # Criando um objeto da classe Image a partir da imagem de referência
    esp_img = Image(esp_data, "niki_lauda")

    # Instanciando histograma da imagem de referência
    esp_hist = hist.img_histogram(esp_img)

    # Exibindo o histograma da imagem de referência
    hist.plot_histogram(esp_hist, "reference_hist")

    # instanciando a resolução da imagem de referência para calculo da lut
    esp_resolution = esp_img.shape[0] * esp_img.shape[1]

    # calculo da "Look up table" da nova imagem a partir da source e da de referência
    img_esp_lut = hist.lut(src_hist, resolution, esp_hist, esp_resolution)

    # equalizando o histograma da imagem original a partir da lut especificada
    img_esp_eq_hist = hist.cdf_equalizer(src_hist, img_esp_lut)

    hist.plot_histogram(img_esp_eq_hist, "result_hist")

    # equalização
    img_eq = eq.cdf_eq_image(img_esp_lut, src_img, "esp_niki")

    src_img.plot_img()
    esp_img.plot_img()
    img_eq.plot_img()
