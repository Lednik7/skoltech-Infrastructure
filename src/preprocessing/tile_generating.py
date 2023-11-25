import os

import cv2
import numpy as np


def add_padding_tile(image, tile_size):
    pad_x = tile_size - image.shape[1]
    pad_y = tile_size - image.shape[0]

    if pad_x > 0 or pad_y > 0:
        return cv2.copyMakeBorder(image, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT)
    return image


def split_image(image, tile_size=256, overlap=0, output_folder=None, mask=None):
    if overlap != 0:
        raise NotImplementedError()
    step_size = int(tile_size * (1 - overlap))

    steps_x = (image.shape[1] + step_size - 1) // step_size
    steps_y = (image.shape[0] + step_size - 1) // step_size

    if output_folder:
        image_dir = f"{output_folder}/images"
        mask_dir = f"{output_folder}/masks" if mask is not None else None
        os.makedirs(image_dir, exist_ok=True)
        if mask_dir:
            os.makedirs(mask_dir, exist_ok=True)

    tiles = []
    for i in range(steps_x):
        for j in range(steps_y):
            x, y = i * step_size, j * step_size
            image_tile = image[y:y + tile_size, x:x + tile_size]
            image_tile = add_padding_tile(image_tile, tile_size)

            if mask is not None:
                mask_tile = mask[y:y + tile_size, x:x + tile_size]
                mask_tile = add_padding_tile(mask_tile, tile_size)

            if output_folder:
                cv2.imwrite(f"{image_dir}/tile_{x}_{y}.png", image_tile)
                if mask is not None:
                    cv2.imwrite(f"{mask_dir}/mask_tile_{x}_{y}.png", mask_tile)
            else:
                tiles.append(
                    (image_tile, mask_tile) if mask is not None else image_tile)
    return tiles


def merge_tiles(tiles, original_size, tile_size=256, overlap=0):
    if overlap != 0:
        raise NotImplementedError()
    image = np.zeros((original_size[0], original_size[1]), dtype=np.float32)
    step_size = int(tile_size * (1 - overlap))

    steps_x = original_size[0] // step_size
    if original_size[0] % step_size != 0:
        steps_x += 1
    steps_y = original_size[1] // step_size
    if original_size[1] % step_size != 0:
        steps_y += 1

    k = 0
    for i in range(steps_y):
        for j in range(steps_x):
            x, y = i * step_size, j * step_size

            tile_width = min(tile_size, original_size[1] - x)
            tile_height = min(tile_size, original_size[0] - y)

            if tile_width <= 0 or tile_height <= 0:
                continue

            tile = tiles[k][:tile_height, :tile_width]
            image[y:y + tile_height, x:x + tile_width] = tile
            k += 1
    return image


if __name__ == '__main__':
    import glob
    import tqdm
    import shutil

    data_path = "../../data/digital_leaders"
    image_folder = os.path.join(data_path, 'images')
    mask_folder = os.path.join(data_path, 'masks')

    image_paths = list(glob.glob(os.path.join(image_folder, '*.png')))
    mask_paths = [image_path.replace("image", "mask") for image_path in image_paths]

    paths = list(zip(image_paths, mask_paths))

    for image_path, mask_path in tqdm.tqdm(paths, total=len(paths)):
        output_folder = os.path.join(data_path, 'tiles',
                                     os.path.splitext(os.path.basename(image_path))[0])
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        split_image(image, tile_size=512,
                    overlap=0, output_folder=output_folder, mask=mask)
