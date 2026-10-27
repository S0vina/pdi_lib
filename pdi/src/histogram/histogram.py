from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from pdi.src.core.image import Image


class Histogram:
    def __init__(self, title):
        pass

    @property
    def histogram_title(self):
        return self._title

    def set_histogram_title(self, new_title: str):
        self._title = new_title

    def img_histogram(self, img: Image) -> np.ndarray:
        if not isinstance(img, Image):
            raise TypeError(
                "Histogram class shoud receive a Image(ndarray type) object for generate a histogram"
            )
        img_hist = np.zeros(256, dtype=int)
        rows, cols = img.shape

        for i in range(rows):
            for j in range(cols):
                img_hist[img.data[i][j]] += 1

        return img_hist

    def cdf(self, hist: np.ndarray) -> np.ndarray:
        """ "Retorna a probabilidade acumulada normalizada dos níveis de intesidade"""
        # depois analisar se inicializo como None para verificar se o usuário quer fazer outra equalização
        # ou apenas inicializo o vetor toda vez e fodase
        cdf = np.zeros(256, dtype=int)
        acumulator = 0

        for i in range(256):
            acumulator += hist[i]
            cdf[i] = acumulator

        return cdf

    def lut(
        self,
        src_hist: np.ndarray,
        src_resolution: int,
        target_hist: Optional[np.ndarray] = None,
        target_resolution: Optional[int] = None,
    ) -> np.ndarray:
        lut = np.zeros(256, dtype=int)

        cdf = self.cdf(src_hist)
        for i in range(256):
            lut[i] = max(0, min(255, round((cdf[i] / src_resolution) * 255)))

        # Se não houver especificada
        if target_hist is None:
            return lut

        elif target_hist is not None:
            target_cdf = self.cdf(target_hist)
            target_lut = np.zeros(256, dtype=int)
            esp_lut = np.zeros(256, dtype=int)

            for i in range(256):
                target_lut[i] = max(
                    0, min(255, round((target_cdf[i] / target_resolution) * 255))
                )

            for i in range(256):
                src_lut_value = lut[i]  # 1,2
                min_value = 256
                min_index = 0

                for j in range(256):
                    diff = abs(src_lut_value - target_lut[j])

                    if diff < min_value:
                        min_value = diff
                        min_index = j

                esp_lut[i] = min_index

            return esp_lut

    def cdf_equalizer(self, src_hist: np.ndarray, lut: np.ndarray) -> np.ndarray:
        equalized_hist = np.zeros(256, dtype=int)

        for i in range(1, 255):
            equalized_hist[lut[i]] = src_hist[i]

        return equalized_hist

    def plot_histogram(self, hist: np.ndarray, title: str) -> None:
        plt.bar(range(256), hist, color="skyblue", edgecolor="black")
        plt.title(f"{title}")
        plt.xlabel("intesity")
        plt.ylabel("pixels")

        plt.show()

    def plot_comparison(
        self,
        hist1: np.ndarray,
        hist2: np.ndarray,
        label1: str = "",
        label2: str = "",
        title: str = "",
    ) -> None:
        """Exibe dois histogramas sobrepostos em um único gráfico.

        :param hist1: Array de 256 elementos do primeiro histograma.
        :param hist2: Array de 256 elementos do segundo histograma.
        :param label1: Rótulo da primeira distribuição no gráfico.
        :param label2: Rótulo da segunda distribuição no gráfico.
        :param title: Título do gráfico.
        """
        bins = np.arange(256)

        plt.figure(figsize=(9, 4.5))

        plt.bar(
            bins,
            hist1,
            color="#2b5c8f",
            alpha=0.55,
            width=1.0,
            label=label1,
            edgecolor="none",
        )

        plt.bar(
            bins,
            hist2,
            color="#ff0002",
            alpha=0.55,
            width=1.0,
            label=label2,
            edgecolor="none",
        )

        plt.title(title, fontsize=12, fontweight="bold", pad=12)
        plt.xlabel("Intensity (0 - 255)", fontsize=10)
        plt.ylabel("Pixels", fontsize=10)
        plt.xlim([0, 255])
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(loc="upper right", frameon=True)

        plt.tight_layout()
        plt.show()

    def normalize_histogram(self):
        pass
