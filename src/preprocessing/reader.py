import os

import cv2
from PIL import Image


class Reader:
    def __init__(self, data_path, read_type="pillow", mode="train"):
        assert read_type in ["opencv", "pillow"]
        self.read_type = read_type
        self.data_path = data_path
        self.image_path = os.path.join(data_path, 'images')
        self.mask_path = os.path.join(data_path, 'masks')
        self.test_image_pattern = mode + '_image_{:03}.png'
        self.test_mask_pattern = mode + '_mask_{:03}.png'

    def get_image_name(self, idx: int):
        return self.test_image_pattern.format(idx)

    def get_mask_name(self, idx: int):
        return self.test_mask_pattern.format(idx)

    def get_image_path(self, idx: int):
        return os.path.join(self.image_path, self.get_image_name(idx))

    def get_mask_path(self, idx: int):
        return os.path.join(self.mask_path, self.get_mask_name(idx))

    def read_mask(self, idx: int):
        if self.read_type == "opencv":
            image = cv2.imread(self.get_mask_path(idx))
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return Image.open(self.get_mask_path(idx))

    def read_image(self, idx: int):
        if self.read_type == "opencv":
            image = cv2.imread(self.get_image_path(idx))
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.open(self.get_image_path(idx))

    def read_sample(self, idx: int):
        return self.read_image(idx), self.read_mask(idx)

    def __str__(self):
        custom_string = f'Image path: {self.image_path}\n'
        custom_string += f'Mask path: {self.mask_path}'
        return custom_string
