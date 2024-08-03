__all__ = [
    'convert_rgb',
]

from PIL import Image


def convert_rgb(image: Image.Image) -> Image.Image:
    if image.format == 'PNG':
        image = image.convert('RGBA')
        image.load()
        *_, alpha = image.split()
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=alpha)
        return background
    return image.convert('RGB')
