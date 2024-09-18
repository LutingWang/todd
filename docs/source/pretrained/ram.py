import torch
import torchvision.transforms.v2 as tf_v2
from einops.layers.torch import Rearrange

import todd.tasks.image_classification as ic
from todd.datasets import IMAGENET_MEAN, IMAGENET_STD
from todd.tasks.image_classification.models.ram import Categories
from todd.utils import get_image

url = (  # pylint: disable=invalid-name
    'https://raw.githubusercontent.com/OPPOMKLab/recognize-anything/main/'
    'images/demo/demo1.jpg'
)
image = get_image(url)

transforms = tf_v2.Compose([
    torch.from_numpy,
    Rearrange('h w c -> 1 c h w'),
    tf_v2.Resize((384, 384)),
    tf_v2.ToDtype(torch.float32, True),
    tf_v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
tensor = transforms(image)

categories = Categories.load()

ram_plus = ic.models.RAMplus(num_categories=len(categories))
ram_plus.load_pretrained('pretrained/ram/ram_plus_swin_large_14m.pth')
ram_plus.requires_grad_(False)
ram_plus.eval()

logits = ram_plus(tensor)

preds = categories.decode(logits)

# yapf: disable
targets = [[
    'armchair', 'blanket', 'lamp', 'carpet', 'couch', 'dog', 'gray', 'green',
    'hassock', 'home', 'lay', 'living room', 'picture frame', 'pillow',
    'plant', 'room', 'wall lamp', 'sit', 'wood floor'  # noqa: C812 E501 pylint: disable=line-too-long
]]
# yapf: enable

assert preds == targets
