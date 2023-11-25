from typing import List

import cv2
import numpy as np
from tqdm.notebook import tqdm

from src.modelling.ensemble import Ensemble
from src.modelling.metrics import f1_score
from src.preprocessing.tile_generating import merge_tiles, split_image
from yolov_preproc.contours import remove_mini_house


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
            threshold: float,
            calc_background: bool = False
    ) -> float:
        assert 0 <= threshold <= 1
        predictions = self.predict_many(images)
        total_score = 0
        if calc_background:
            total_bg_score = 0
        for pred, mask in zip(predictions, masks):
            binary_pred = (pred > threshold).astype(np.int16)

            if calc_background:
                bg_mask = 1 - mask
                bg_pred = 1 - binary_pred
                total_bg_score += f1_score(bg_mask, bg_pred) / len(masks)
            total_score += f1_score(mask, binary_pred) / len(masks)
        if calc_background:
            return total_score, total_bg_score
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
        average_mask = (results + shifted_results) / 2
        cleared = remove_mini_house(average_mask, 1000)
        cleared = (cleared > 0.5).astype(np.int16)
        return cleared
