import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2

from src.modelling.ensemble import AbstractModel


class DeeplabMOCO(AbstractModel):
    def __init__(self, weights_path: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model = smp.DeepLabV3(encoder_name="resnet50",
                                   in_channels=3, classes=1,
                                   activation="sigmoid")
        self.model.load_state_dict(torch.load(weights_path))
        self.model.to(self.device)
        self.model.eval()
        self.transforms = A.Compose(
            [
                A.Resize(256, 256),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

    def preprocess(self, image: np.ndarray):
        image = self.transforms(image=image)["image"]
        return image.unsqueeze(0)

    def postprocess(self, x: torch.Tensor) -> np.ndarray:
        x = x.squeeze(0).squeeze(0).detach().cpu().numpy()
        return x

    def predict(self, image: np.ndarray) -> np.ndarray:
        inputs = self.preprocess(image)
        predictions = self.model(inputs.to(self.device))
        return self.postprocess(predictions)
