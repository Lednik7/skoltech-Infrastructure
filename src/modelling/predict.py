from typing import List

import numpy as np
from tqdm.notebook import tqdm

from src.modelling.ensemble import Ensemble
from src.modelling.metrics import f1_score
from src.preprocessing.tile_generating import merge_tiles, split_image


class Predictor:
    def __init__(self, model: Ensemble):
        self.model = model

    def predict_tile(self, image: np.ndarray) -> np.ndarray:
        return self.model.predict(image)

    def predict(self, image: np.ndarray) -> np.ndarray:
        tiles = split_image(image, tile_size=512, overlap=0)
        predictions = [self.predict_tile(tile) for tile in tqdm(tiles)]
        return merge_tiles(predictions, image.shape[:2], tile_size=512, overlap=0)

    def predict_many(self, images: List[np.ndarray]) -> List[np.ndarray]:
        return [self.predict(image) for image in images]

    def validate(
            self,
            images: List[np.ndarray],
            masks: List[np.ndarray],
            threshold: float
    ) -> float:
        assert 0 <= threshold <= 1
        predictions = self.predict_many(images)
        total_score = 0
        for pred, mask in zip(predictions, masks):
            binary_pred = (pred > threshold).astype(np.int16)
            f1_score(mask, binary_pred)
            total_score += f1_score(mask, binary_pred) / len(masks)
        return total_score
