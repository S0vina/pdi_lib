from typing import List, Self, Tuple
import numpy as np


class Mask:
    LOW_PASS = "low_pass"
    GAUSSIAN = "gaussian"
    LAPLASSIAN = "laplassian"
    FOUR_NEIGHBOR = "4_neighbor"
    EIGHT_NEIGHBOR = "8-neighbor"

    def __init__(self, array, mt, mn, h: int, w: int):
        self.array = array
        self.mask_type = mt
        self.mask_neighbor = mn
        self.height = h
        self.width = w

    @property
    def shape(self):
        return (self.height, self.width)

    @classmethod
    def any_shape(
        cls,
        mask_type,
        mask_neighbor=None,
        height: int = -1,
        width: int = -1,
        weight: float = 0,
    ) -> Self:
        if height < 0 or width < 0:
            raise ValueError("height and width values should be greater than 0")

        array = np.full((height, width), weight)

        return cls(array, mask_type, mask_neighbor, height, width)

    @classmethod
    def from_type(cls, mask_type, mask_neighbor, n: int = 3) -> Self:
        if mask_type is None:
            raise ValueError("mask type should be specified")

        if mask_neighbor is None:
            raise ValueError("should specify what neighborhood is taken")

        if n % 2 != 1:
            raise ValueError("a pair mask is not implemented in this lib")

        if mask_type == cls.LOW_PASS:
            center = n // 2

            if mask_neighbor == cls.FOUR_NEIGHBOR:
                weight = 1.0 / (2 * n - 1)
                kernel = cls.any_shape(mask_type, mask_neighbor, n, n, 0)
                kernel_h, kernel_w = kernel.shape

                for i in range(kernel_h):
                    for j in range(kernel_w):
                        if i == center or j == center:
                            kernel.array[i][j] = weight

                return kernel

            if mask_neighbor == cls.EIGHT_NEIGHBOR:
                weight = 1.0 / (n * n)
                kernel = cls.any_shape(mask_type, mask_neighbor, n, n, weight)

                return kernel

        if mask_type == cls.GAUSSIAN:
            pass

        if mask_type == cls.LAPLASSIAN:
            instance = cls.any_shape(mask_type, mask_neighbor, n, n, 1)
            center_index = n // 2

            # laplassian with 4 neighbor
            if mask_neighbor == cls.FOUR_NEIGHBOR:
                instance.array[center_index][center_index] = -4

                for i in range(n):
                    for j in range(n):
                        if i == center_index and j == center_index:
                            continue

                        if i == center_index or j == center_index:
                            instance.array[i][j] = 0

            if mask_neighbor == cls.EIGHT_NEIGHBOR:
                instance.array[center_index][center_index] = -8

            return instance

        raise ValueError(f"Unknown mask type: {mask_type}")

    def it_fits(self, i: int, j: int, img_t: Tuple[int, int]) -> bool:
        img_h, img_w = img_t
        mask_h, mask_w = self.shape  # ou self.height, self.width

        h_half = mask_h // 2
        w_half = mask_w // 2

        # Verifica limites superiores e esquerdos
        if i - h_half < 0 or j - w_half < 0:
            return False

        # Verifica limites inferiores e direitos
        if i + h_half >= img_h or j + w_half >= img_w:
            return False

        return True

    def dot_product(
        self, i: int, j: int, img_data: np.ndarray, img_shape: Tuple[int, int]
    ):

        # Calcula o deslocamento (offset) do centro
        h_half = self.height // 2
        w_half = self.width // 2

        if self.it_fits(i, j, img_shape):
            roi = img_data[i - h_half : i + h_half + 1, j - w_half : j + w_half + 1]

            if roi.shape == (self.height, self.width):
                val = round(min(255, max(0, np.sum(roi * self.array))))
                return int(val)

            return -1

        return img_data[i][j]

    def get_list(self, center_i: int, center_j: int, img_data: np.ndarray) -> List:
        mask_h, mask_w = self.shape
        flat_roi = []

        half_h = mask_h // 2
        half_w = mask_w // 2

        for r in range(center_i - half_h, center_i + half_h + 1):
            for c in range(center_j - half_w, center_j + half_w + 1):
                flat_roi.append(img_data[r][c])

        return flat_roi

    def get_local_mean(self, i: int, j: int, img_data:np.ndarray):
        mask_h, mask_w = self.shape
        roi = img_data[]

    def get_local_variance(self):
        pass

    @classmethod
    def flip(cls, mask: Self) -> Self:
        mask_h, mask_w = mask.shape
        array = np.zeros((mask_h, mask_w))
        for i in range(mask_h):
            for j in range(mask_w):
                array[i][j] = mask.array[mask_h - i - 1][mask_w - j - 1]

        return cls(array, mask.mask_type, mask.mask_neighbor, mask_h, mask_w)
