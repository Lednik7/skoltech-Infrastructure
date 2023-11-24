import os.path
import sys

import cv2
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
    convert_tensor_to_numpy,
    adjust_model,)
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


def extract(original_image):
    original_height, original_width = original_image.shape[:2]

    if (original_height, original_width) != (MAX_SIZE, MAX_SIZE):
        original_image = handle_image_size(original_image, (MAX_SIZE, MAX_SIZE))

    # Apply Normalization
    normalized_image = min_max_image_net(img=original_image)

    tensor_image = add_extra_dimension(to_input_image_tensor(normalized_image))

    with torch.no_grad():
        # Perform prediction
        prediction = TRAINED_MODEL(tensor_image)
        prediction = prediction.sigmoid()

    prediction_binary = convert_tensor_to_numpy(prediction[0]).reshape(
        (MAX_SIZE, MAX_SIZE)
    )

    prediction_3_channels = cv2.cvtColor(prediction_binary, cv2.COLOR_GRAY2RGB)

    dst = cv2.addWeighted(
        original_image,
        1,
        (prediction_3_channels * (0, 255, 0)).astype(np.uint8),
        0.4,
        0,
    )
    return prediction_binary, prediction_3_channels, dst
def predict(image):
    prediction_binary, prediction_3_channels, dst = extract(image)
    return prediction_binary, prediction_3_channels, dst


class FootPrintModel(AbstractModel):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.model = TRAINED_MODEL

        state_dict = model_zoo.load_url(MODEL_URL, progress=True, map_location="cpu")
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
        return image.unsqueeze(0)

    def postprocess(self, x: torch.Tensor) -> np.ndarray:
        x = x.squeeze(0).squeeze(0).detach().cpu().numpy()
        return x

    def predict(self, image: np.ndarray) -> np.ndarray:
        prediction_binary, prediction_3_channels, dst = extract(image)
        return prediction_binary
