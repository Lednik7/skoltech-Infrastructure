import os
from itertools import product

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None


def add_padding_tile(image, size):
    pad_x = size[0] - image.size[0]
    pad_y = size[1] - image.size[1]

    if pad_x > 0 or pad_y > 0:
        padding = (0, 0, pad_x, pad_y)
        return ImageOps.expand(image, padding)
    return image


def split_image(image_path, mask_path, output_folder, tile_size=(256, 256), overlap=0):
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    step_size = int(tile_size[0] * (1 - overlap))

    image_dir = f"{output_folder}/images"
    mask_dir = f"{output_folder}/masks"
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for i, j in product(range(0, image.width - tile_size[0] + 1, step_size),
                        range(0, image.height - tile_size[1] + 1, step_size)):
        box = (i, j, i + tile_size[0], j + tile_size[1])
        image_tile = image.crop(box)
        mask_tile = mask.crop(box)

        image_tile = add_padding_tile(image_tile, tile_size)
        mask_tile = add_padding_tile(mask_tile, tile_size)

        image_tile.save(f"{image_dir}/tile_{i}_{j}.png")
        mask_tile.save(f"{mask_dir}/mask_tile_{i}_{j}.png")


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
        outer_folder = os.path.join(data_path, 'tiles',
                                    os.path.splitext(os.path.basename(image_path))[0])
        if os.path.exists(outer_folder):
            shutil.rmtree(outer_folder)
        os.makedirs(outer_folder)

        split_image(image_path, mask_path, outer_folder, tile_size=(512, 512),
                    overlap=0)
