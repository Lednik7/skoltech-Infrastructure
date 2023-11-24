import numpy as np

from src.modelling.ensemble import Ensemble
from src.preprocessing.tile_generating import merge_tiles, split_image


class Predictor:
    def __init__(self, model: Ensemble):
        self.model = model

    def predict_tile(self, image: np.ndarray) -> np.ndarray:
        return self.model.predict(image)

    def predict(self, image: np.ndarray) -> np.ndarray:
        tiles = split_image(image, tile_size=512, overlap=0)
        predictions = [self.predict_tile(tile) for tile in tiles]
        return merge_tiles(predictions, image.shape[:2], tile_size=512, overlap=0)
