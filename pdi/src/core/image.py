from pathlib import Path
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import cv2


class Image:
    def __init__(self, data: np.ndarray, name: str):
        pass

    @classmethod
    def get_matrix(cls, path: Union[str, Path]) -> np.ndarray:
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            arr = np.asarray(img)

            return arr

        except Exception as e:
            raise IOError(f"The image opened on {path} has an issue: {e}")

    @classmethod
    def save(cls, matrix: np.ndarray, path: Union[str, Path]) -> None:
        cv2.imwrite(path, matrix)

    @classmethod
    def plot(cls, matrix: np.ndarray):
        plt.imshow(matrix, cmap="gray")
        plt.show()
