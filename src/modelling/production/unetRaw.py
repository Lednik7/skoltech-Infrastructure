import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2

from src.modelling.ensemble import AbstractModel


class UnetRaw(AbstractModel):
    def __init__(self, weights_path: str, device: str = "cuda"):
        super().__init__()
        self.model = smp.Unet(encoder_name="efficientnet-b0",
                              in_channels=3, classes=1,
                              activation="sigmoid")
        self.model.load_state_dict(torch.load(weights_path))
        self.model.to(device)
        self.transforms = A.Compose(
            [
                A.Resize(256, 256),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

    def preprocess(self, image: np.ndarray):
        return self.transforms(image=image)["image"]

    def postprocess(self, x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    def predict(self, image: np.ndarray) -> np.ndarray:
        inputs = self.preprocess(image)
        predictions = self.model(inputs)
        return self.postprocess(predictions)
