import os

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None


def add_padding_tile(image, size):
    pad_x = size - image.size[0]
    pad_y = size - image.size[1]

    if pad_x > 0 or pad_y > 0:
        padding = (0, 0, pad_x, pad_y)
        return ImageOps.expand(image, padding)
    return image


def split_image(image: Image, mask: Image, tile_size=256, overlap=0,
                output_folder=None):
    if overlap != 0:
        raise NotImplementedError()
    step_size = int(tile_size * (1 - overlap))

    if output_folder is not None:
        image_dir = f"{output_folder}/images"
        mask_dir = f"{output_folder}/masks"
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

    steps_x = image.size[0] // step_size
    if image.size[0] % step_size != 0:
        steps_x += 1
    steps_y = image.size[1] // step_size
    if image.size[1] % step_size != 0:
        steps_y += 1

    for i in range(steps_x):
        for j in range(steps_y):
            left_top = (i * step_size, j * step_size)
            right_bottom = (left_top[0] + tile_size, left_top[1] + tile_size)

            image_tile = image.crop(left_top + right_bottom)
            mask_tile = mask.crop(left_top + right_bottom)

            image_tile = add_padding_tile(image_tile, tile_size)
            mask_tile = add_padding_tile(mask_tile, tile_size)

            if output_folder is not None:
                image_tile.save(f"{image_dir}/tile_{left_top[0]}_{left_top[1]}.png")
                mask_tile.save(f"{mask_dir}/mask_tile_{left_top[0]}_{left_top[1]}.png")
            else:
                yield image_tile, mask_tile


def merge_tiles(tiles: list, original_size, tile_size=256, overlap=0):
    if overlap != 0:
        raise NotImplementedError()
    image = Image.new('L', original_size)
    step_size = int(tile_size * (1 - overlap))

    steps_x = original_size[0] // step_size
    if original_size[0] % step_size != 0:
        steps_x += 1
    steps_y = original_size[1] // step_size
    if original_size[1] % step_size != 0:
        steps_y += 1

    k = 0
    for i in range(steps_x):
        for j in range(steps_y):
            left_top = (i * step_size, j * step_size)
            right_bottom = (left_top[0] + tile_size, left_top[1] + tile_size)
            image.paste(tiles[k], left_top + right_bottom)
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

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        split_image(image, mask, tile_size=512,
                    overlap=0, output_folder=output_folder)
