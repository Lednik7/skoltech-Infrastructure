import os
import cv2
import numpy as np

from skimage import measure

import shapely
from shapely.geometry import Polygon, MultiPolygon

from tqdm.notebook import tqdm
import rasterio.features

def create_geometry(mask: np.array,
                    pixel_idx: int = 1) -> shapely.geometry:
    # polypic = (mask == pixel_idx) * 2
    polypic = mask * 2

    # find contours
    # Not sure why 1.0 works as a level -- maybe experiment with lower values
    contours = measure.find_contours(polypic, 1.0)

    pols = []

    for idx, contour in enumerate(contours):
        pol = Polygon(np.flip(contours[idx] * 1, 1))  # .simplify(SIMPLIFY)
        pols.append(pol)
    adj = {}
    for i in range(len(pols)):
        for j in range(i + 1, len(pols)):
            if pols[i].intersects(pols[j]):
                adj[i] = adj.get(i, {i}) | {j}
    adj_un = []
    for i in adj:
        for idx in range(len(adj_un)):
            if i in adj_un[idx]:
                adj_un[idx] = adj_un[idx] | adj[i] | {i}
                break
        else:
            adj_un.append(adj[i] | {i})
    res_pols = [[] for _ in adj_un]
    for pol_id, pol in enumerate(pols):
        for idx, group in enumerate(adj_un):
            if pol_id in group:
                res_pols[idx].append(pol)
                break
        else:
            res_pols.append(pol)
    for i in range(len(adj_un)):
        res_pols[i] = MultiPolygon(res_pols[i])

    return res_pols


def create_contour(mask: np.array,
                   pixel_idx: int = 1,
                   smooth: bool = False) -> dict:
    mask = (mask == pixel_idx) * 1

    mask = np.array(mask, dtype=np.uint8)
    #
    if smooth:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, cv2.BORDER_REFLECT)

    kernel = np.ones((5, 5), np.float32) / 25
    mask = cv2.filter2D(mask, -1, kernel)

    # mask = cv2.GaussianBlur(mask, (5, 5), 0)

    organ_contours = {}

    for ci, i in enumerate(np.unique(mask)[1:]):
        res_pol = create_geometry(mask, pixel_idx=pixel_idx)
        organ_contours[i] = res_pol

    return organ_contours


def remove_mini_house(mask, trch=1):
    new_mask = np.zeros(mask.shape)
    contours = create_contour(mask)
    for contour in tqdm(contours[1]):
        if contour.area < trch:
            print('remove')
            continue

        new_mask += rasterio.features.rasterize([contour], out_shape=mask.shape)
        # new_mask += calculate_area(contour, mask.shape)
        #
    return new_mask

