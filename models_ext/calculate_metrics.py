from src.modelling.metrics import DiceMetric
import matplotlib.pyplot as plt
import numpy as np
import cv2

import torch
def plot_result_tile(image, pred_mask, true_mask):
    f1_loss = DiceMetric()

    _, axs = plt.subplots(1, 3, figsize=(12, 12))

    axs[0].imshow(image)
    axs[0].set_title('Image')

    # kernel = np.ones((6, 6), np.uint8)
    #
    # # Using cv2.erode() method
    # pred_mask = cv2.erode(pred_mask, kernel, cv2.BORDER_REFLECT)

    axs[1].imshow(pred_mask * 1000)
    axs[1].set_title(f'Pred mask - f1: {np.round(f1_loss(torch.Tensor(pred_mask), torch.Tensor(true_mask)), 2)}')


    axs[2].imshow(true_mask * 1000)
    axs[2].set_title('True mask')
