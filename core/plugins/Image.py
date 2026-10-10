from pathlib import Path
from typing import Tuple, Union
import numpy as np
import cv2


class Image:
    def __init__(self, data: np.ndarray, name: str):
        if not isinstance(data, np.ndarray):
            raise TypeError("Os dados da imagem devem ser um numpy.ndarray")

        self._data = data
        self._img_name = name

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def name(self) -> str:
        return self._img_name

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

    @property
    def is_grayscale(self) -> bool:
        return True

    @classmethod
    def get_data_from_file(cls, path: Union[str, Path]) -> np.ndarray:
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            arr = np.asarray(img)

            return arr

        except Exception as e:
            raise IOError(f"The image opened on {path} has an issue: {e}")

    @classmethod
    def save_img(cls, self, path: Union[str, Path]) -> None:
        cv2.imwrite(path, self._data)

    def plot_img(self):
        while True:
            cv2.imshow(self.name, self.data)

            if cv2.waitKey(0) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

    def __repr__(self) -> str:
        return f"<Image shape={self.shape} dtype={self._data.dtype}>"
