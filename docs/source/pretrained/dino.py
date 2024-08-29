import torch
import torchvision.transforms.v2 as tf_v2
from einops.layers.torch import Rearrange

from todd.datasets import IMAGENET_MEAN, IMAGENET_STD, COCODataset
from todd.models.modules import DINO, DINOv2
from todd.utils import get_image

url = COCODataset.url('val', 2017, 39769)
image = get_image(url)

transforms = tf_v2.Compose([
    torch.from_numpy,
    Rearrange('h w c -> 1 c h w'),
    tf_v2.Resize(256, tf_v2.InterpolationMode.BICUBIC),
    tf_v2.CenterCrop(256),
    tf_v2.ToDtype(torch.float32, scale=True),
    tf_v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
tensor = transforms(image)

dino = DINO()
dino.load_pretrained('pretrained/dino/dino_vitbase16_pretrain.pth')
cls_, x = dino(tensor, False)
assert torch.allclose(
    cls_[:, :3],
    torch.tensor([[2.3480, -5.4728, 3.2335]]),
    atol=1e-4,
)
assert torch.allclose(
    x[:, :3, :3],
    torch.tensor([[
        [-1.5503, -2.4765, 1.1179],
        [-2.6262, -1.3692, -0.1952],
        [-2.4387, 0.5866, 0.0967],
    ]]),
    atol=1e-4,
)

dinov2 = DINOv2(
    patch_size=14,
    patch_wh=(37, 37),
    width=1024,
    depth=24,
    num_heads=16,
)
dinov2.load_pretrained('pretrained/dino/dinov2_vitl14_pretrain.pth')
cls_, x = dinov2(tensor, False)
assert torch.allclose(
    cls_[:, :3],
    torch.tensor([[-2.7620, -2.1128, 0.4863]]),
    atol=1e-4,
)
assert torch.allclose(
    x[:, :3, :3],
    torch.tensor([[
        [2.4025, -1.0418, 1.6840],
        [1.6887, 0.0430, 3.5852],
        [0.5082, -0.5240, 0.8340],
    ]]),
    atol=1e-4,
)
