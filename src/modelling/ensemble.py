from typing import Dict, Optional

import numpy as np

from src.modelling import AbstractModel


class Ensemble:
    def __init__(self, models: Dict[str, AbstractModel]):
        self.models = models

    def predict_single(self, image: np.ndarray,
                       model_name: Optional[str] = None) -> np.ndarray:
        return self.models[model_name].predict(image)

    def predict(self, image: np.ndarray, return_average: bool = True) -> np.ndarray:
        predictions = [self.predict_single(image, model_name) for model_name in
                       self.models]
        if return_average:
            return np.mean(np.stack(predictions), axis=0)
        return np.stack(predictions)

    def __call__(self, image: np.ndarray, return_average: bool = True) -> np.ndarray:
        return self.predict(image, return_average)
