import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2

from src.modelling import AbstractModel


class unetPlusPlusMOCO(AbstractModel):
    def __init__(self, weights_path: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model = smp.UnetPlusPlus(encoder_name="resnet50", encoder_weights=None,
                             in_channels=weights.meta["in_chans"], classes=1,
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
        self.post_transforms = A.Compose(
            [
                A.Resize(512, 512),
            ]
        )

    def preprocess(self, image: np.ndarray):
        image = self.transforms(image=image)["image"]
        return image.unsqueeze(0)

    def postprocess(self, x: torch.Tensor) -> np.ndarray:
        x = x.squeeze(0).squeeze(0).detach().cpu().numpy()
        x = self.post_transforms(image=x)["image"]
        return x

    def predict(self, image: np.ndarray) -> np.ndarray:
        inputs = self.preprocess(image)
        predictions = self.model(inputs.to(self.device))
        return self.postprocess(predictions)


class unetPlusPlusMOCOEnsemble(AbstractModel):
    def __init__(self, weights_paths: list, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.models = []
        for weights_path in weights_paths:
            self.models.append(unetPlusPlusMOCO(weights_path=weights_path, device=device))

    def predict(self, image: np.ndarray) -> np.ndarray:
        outputs = []
        for model in self.models:
            outputs.append(model.predict(image))
        return np.mean(outputs)
