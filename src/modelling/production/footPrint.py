import os.path
import sys

import numpy as np
import torch
from torch.utils import model_zoo

module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from building_footprint_segmentation.seg.binary.models import ReFineNet
from building_footprint_segmentation.helpers.normalizer import min_max_image_net
from building_footprint_segmentation.utils.py_network import (
    to_input_image_tensor,
    add_extra_dimension,
    adjust_model, )
from building_footprint_segmentation.utils.operations import handle_image_size

from src.modelling import AbstractModel

import albumentations as A

import warnings

warnings.filterwarnings("ignore")

MAX_SIZE = 512
TRAINED_MODEL = ReFineNet()
MODEL_URL = "https://github.com/fuzailpalnak/building-footprint-segmentation/releases/download/alpha/refine.zip"


def set_model_weights():
    state_dict = model_zoo.load_url(MODEL_URL, progress=True, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    TRAINED_MODEL.load_state_dict(adjust_model(state_dict))


class FootPrintModel(AbstractModel):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.model = TRAINED_MODEL

        state_dict = model_zoo.load_url(MODEL_URL, progress=True, map_location=device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self.model.load_state_dict(adjust_model(state_dict))
        self.model.to(self.device)
        self.model.eval()
        self.transforms = A.Compose(
            [
                A.Resize(512, 512),
            ]
        )

    def preprocess(self, image: np.ndarray):
        image = self.transforms(image=image)["image"]

        original_height, original_width = image.shape[:2]

        if (original_height, original_width) != (MAX_SIZE, MAX_SIZE):
            image = handle_image_size(image, (MAX_SIZE, MAX_SIZE))

        # Apply Normalization
        normalized_image = min_max_image_net(img=image)

        tensor_image = add_extra_dimension(to_input_image_tensor(normalized_image))
        return tensor_image.to(self.device)

    def postprocess(self, x: torch.Tensor) -> np.ndarray:
        x = x.squeeze(0).squeeze(0).detach().cpu().numpy()
        return x

    def predict(self, image: np.ndarray) -> np.ndarray:
        image = self.preprocess(image)

        with torch.no_grad():
            predictions = TRAINED_MODEL(image)
            predictions = predictions.sigmoid()

        prediction_binary = self.postprocess(predictions)
        return prediction_binary
