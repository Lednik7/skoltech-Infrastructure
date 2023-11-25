from typing import Dict, List

import numpy as np

from src.modelling import AbstractModel


class Ensemble:
    def __init__(self, models: Dict[str, AbstractModel], weights: List[float] = None):
        if weights is None:
            weights = [1 / len(models) for _ in models]
        assert len(weights) == len(models)
        assert np.allclose(sum(weights), 1)
        self.weights = weights
        self.models = models

    def predict(self, image: np.ndarray) -> np.ndarray:
        predictions = [self.models[model_name].predict(image) for model_name in
                       self.models]
        shapes = [prediction.shape for prediction in predictions]
        assert len(set(shapes)) == 1, f"Predictions have different shapes: {shapes}"
        return np.average(predictions, axis=0, weights=self.weights)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.predict(image)
