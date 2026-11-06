from pdi.src.core.mask import Mask
from typing import Self, Tuple
import numpy as np


class StructElement(Mask):
    # Variaveis globais vão ser definidas dessa forma
    STAR = "star"

    def __init__(self, array, mt, mn, h: int, w: int) -> None:
        super().__init__(array, mt, mn, h, w)

    @classmethod
    def any_shape(
        cls,
        mask_type,
        mask_neighboor=None,
        height: int = -1,
        width: int = -1,
        weight: float | bool = True,
    ) -> Self:

        if height < 0 or width < 0:
            raise ValueError("height and width values should be greater than 0")

        bool_weight = bool(weight)
        array = np.full((height, width), bool_weight, dtype=bool)

        return cls(array, mask_type, mask_neighboor, height, width)

    @classmethod
    def from_type(cls, mask_type, mask_neighbor, n: int = 3) -> Self:
        if mask_type is None:
            raise ValueError("mask type should be specified")

        if mask_neighbor is None:
            raise ValueError("should specify what neighborhood is taken")

        if n % 2 != 1:
            raise ValueError("a pair mask is not implemented in this lib")

        if mask_type == cls.STAR:
            se = cls.any_shape(mask_type, mask_neighbor, 3, 3)

            for i in range(3):
                for j in range(3):
                    pass

    # Kernel mapping for Dilatation operation
    def _hits(
        self, center_i, center_j, img_matrix: np.ndarray, img_shape: Tuple[int, int]
    ) -> bool:
        img_h, img_w = img_shape
        mask_h, mask_w = self.shape  # ou self.height, self.width

        h_half = mask_h // 2
        w_half = mask_w // 2

        for i in range(mask_h):
            for j in range(mask_w):
                mask_val = self.array[i][j]

                if not mask_val:
                    continue

                img_r = center_i + (i - h_half)
                img_c = center_j + (j - w_half)

                # regiões fora de borda como 0
                if img_r < 0 or img_r >= img_h or img_c < 0 or img_c >= img_w:
                    pixel_val = 0
                else:
                    pixel_val = img_matrix[img_r][img_c]

                if pixel_val == 0:
                    return True

        return False

    def _fits(
        self, center_i, center_j, img_matrix: np.ndarray, img_shape: Tuple[int, int]
    ) -> bool:
        img_h, img_w = img_shape
        mask_h, mask_w = self.shape  # ou self.height, self.width

        h_half = mask_h // 2
        w_half = mask_w // 2

        for i in range(mask_h):
            for j in range(mask_w):
                mask_val = self.array[i][j]

                if not mask_val:
                    continue

                img_r = center_i + (i - h_half)
                img_c = center_j + (j - w_half)

                # regiões fora de borda como 0
                if img_r < 0 or img_r >= img_h or img_c < 0 or img_c >= img_w:
                    pixel_val = 255
                else:
                    pixel_val = img_matrix[img_r][img_c]

                if pixel_val != 0:
                    return False

        return True

    # TALVEZ NÃO SIRVA MAIS PARA PORRA NENHUMA. JOGAR FORA DPS
    def it_fits(
        self,
        center_i: int,
        center_j: int,
        img_t: Tuple[int, int],
        img_array: np.ndarray = np.zeros((1, 1)),
        dilatation: bool = False,
    ) -> bool:

        img_h, img_w = img_t
        mask_h, mask_w = self.shape  # ou self.height, self.width

        h_half = mask_h // 2
        w_half = mask_w // 2

        for i in range(mask_h):
            for j in range(mask_w):
                # CAPTURA DOS VALORES DA IMAGEM DE ENTRADA E MASCARA
                mask_val = self.array[i][j]

                if not mask_val:  # Equivalente a mask_val == 0 ou False
                    continue
                # posicao real na imagem
                img_r = center_i + (i - h_half)
                img_c = center_j + (j - w_half)

                # regiões fora de borda como 0
                if img_r < 0 or img_r >= img_h or img_c < 0 or img_c >= img_w:
                    pixel_val = 255
                else:
                    pixel_val = img_array[img_r][img_c]

                # AQUI É A DEFINIÇÃO DE SE O ELEMENTO ESTRUTURANTE CABE
                if dilatation:
                    if pixel_val == 0:
                        return True
                else:
                    # Se for erosão: se algum ponto ativo falhar e não for 1, rejeita.
                    if pixel_val != 0:
                        return False

        return False if dilatation else True
