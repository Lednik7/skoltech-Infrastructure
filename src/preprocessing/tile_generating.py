import os
from itertools import product

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def add_padding(image, tile_size, overlap):
    pad_x = int(tile_size[0] * (1 - overlap) / 2)
    pad_y = int(tile_size[1] * (1 - overlap) / 2)

    new_width = image.width + 2 * pad_x
    new_height = image.height + 2 * pad_y

    padded_image = Image.new('RGB', (new_width, new_height))
    padded_image.paste(image, (pad_x, pad_y))

    return padded_image


def split_image(image_path, mask_path, output_folder, tile_size=(256, 256), overlap=0):
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    image = add_padding(image, tile_size, overlap)
    mask = add_padding(mask, tile_size, overlap)

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

        image_tile.save(f"{image_dir}/tile_{i}_{j}.png")
        mask_tile.save(f"{mask_dir}/mask_tile_{i}_{j}.png")


def merge_tiles(input_folder, original_size, tile_size=(256, 256), overlap=0):
    step_size = int(tile_size[0] * (1 - overlap))
    pad_x = int(tile_size[0] * (1 - overlap) / 2)
    pad_y = int(tile_size[1] * (1 - overlap) / 2)

    new_width = original_size[0] + 2 * pad_x
    new_height = original_size[1] + 2 * pad_y

    merged_image = Image.new('RGB', (new_width, new_height))

    for i, j in product(range(0, new_width - tile_size[0] + 1, step_size),
                        range(0, new_height - tile_size[1] + 1, step_size)):
        try:
            tile = Image.open(f"{input_folder}/images/tile_{i}_{j}.png")
            merged_image.paste(tile, (i, j), tile)
        except FileNotFoundError:
            print(f"No tile found for position ({i}, {j})")
            continue

    # Crop out the padding
    merged_image = merged_image.crop(
        (pad_x, pad_y, original_size[0] + pad_x, original_size[1] + pad_y))

    return merged_image


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
