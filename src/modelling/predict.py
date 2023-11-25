from typing import List

import cv2
import numpy as np
import torch
import ttach as tta
from albumentations.pytorch import ToTensorV2
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


class TTAPredictor(Predictor):
    basic_tta = tta.Compose([
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        # tta.Rotate90(angles=[0, 90, 180, 270]),
    ])

    def __init__(self, model: Ensemble, tta_transforms=basic_tta):
        super().__init__(model)
        self.tta_transforms = tta_transforms

    def predict(self, image: np.ndarray) -> np.ndarray:
        preds = []
        for i, transformer in enumerate(self.tta_transforms):
            image = ToTensorV2(p=1.0)(image=image)["image"]
            image = image.unsqueeze(0)
            image = transformer.augment_image(image)
            image = image.squeeze(0)
            image = image.numpy().transpose(1, 2, 0)

            cv2.imwrite(f'test_{i}.png', image)

            pred = super().predict(image)

            pred = torch.from_numpy(pred)
            pred = pred.unsqueeze(0).unsqueeze(0)
            pred = transformer.deaugment_mask(pred)
            pred = pred.squeeze(0).squeeze(0).numpy()
            preds.append(pred)

            print(image.shape, pred.shape)
        return np.mean(preds, axis=0)
