from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import io, color
import numpy as np


class Avaliate:
    def __init__(self) -> None:
        pass

    def avaliar(self, img_ref, img_proc, label=""):
        """
        img_ref  : imagem original (uint8, cinza ou RGB)
        img_proc : imagem após filtragem (uint8, cinza ou RGB)
        """
        p = psnr(img_ref, img_proc, data_range=255)

        is_color = img_ref.ndim == 3
        s = ssim(
            img_ref, img_proc, data_range=255, channel_axis=-1 if is_color else None
        )

        return p, s
