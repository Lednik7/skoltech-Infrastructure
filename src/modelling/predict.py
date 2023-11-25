from typing import List

import numpy as np
from tqdm.notebook import tqdm
import cv2

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


def create_shift(image: np.ndarray, tile_size: int) -> np.ndarray:
    shifted_image = cv2.copyMakeBorder(image, tile_size // 2, 0, tile_size // 2, 0,
                                       cv2.BORDER_CONSTANT)
    return shifted_image


class ShiftedPredictor(Predictor):
    def predict(self, image: np.ndarray) -> np.ndarray:
        tile_size = 512
        tiles = split_image(image, tile_size=tile_size, overlap=0)
        predictions = [self.predict_tile(tile) for tile in tqdm(tiles)]
        results = merge_tiles(predictions, image.shape[:2], tile_size=tile_size,
                              overlap=0)

        shifted_image = create_shift(image, tile_size)
        shifted_tiles = split_image(shifted_image, tile_size=tile_size,
                                    overlap=0)
        shifted_predictions = [self.predict_tile(tile) for tile in tqdm(shifted_tiles)]
        shifted_results = merge_tiles(shifted_predictions, shifted_image.shape[:2],
                                      tile_size=tile_size,
                                      overlap=0)
        shifted_results = shifted_results[tile_size // 2:, tile_size // 2:]
        assert results.shape == shifted_results.shape
        return (results + shifted_results) / 2
