from typing import Any

import numpy as np
import torch


class AbstractModel:
    def __init__(self):
        self.model = ...
        pass

    def preprocess(self, image: np.ndarray) -> Any[np.ndarray, torch.Tensor]:
        pass

    def postprocess(self, x) -> np.ndarray:
        pass

    def predict(self, image: np.ndarray) -> np.ndarray:
        inputs = self.preprocess(image)
        predictions = self.model(inputs)
        return self.postprocess(predictions)
