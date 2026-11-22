from pathlib import Path
from typing import Union
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import cv2


class Image:
    def __init__(self, data: np.ndarray, name: str):
        pass

    @classmethod
    def get_data(cls, path: Union[str, Path]) -> np.ndarray:
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            arr = np.asarray(img)

            return arr

        except Exception as e:
            raise IOError(f"The image opened on {path} has an issue: {e}")

    @classmethod
    def save(cls, img: np.ndarray, path: Union[str, Path]) -> None:
        cv2.imwrite(path, img)

    @classmethod
    def plot(cls, img: np.ndarray):
        plt.imshow(img, cmap="gray")
        plt.show()

    @classmethod
    def compare(cls, img_one: np.ndarray, img_two: np.ndarray):
        fig, ax = plt.subplot_mosaic([["img_one", "img_two"]], figsize=(7, 3.5))

        ax["img_one"].imshow(img_one, cmap="gray")
        ax["img_one"].set_title("img_one")
        ax["img_one"].axis("off")

        ax["img_two"].imshow(img_two, cmap="gray")
        ax["img_two"].set_title("img_two")
        ax["img_two"].axis("off")

        plt.show()
