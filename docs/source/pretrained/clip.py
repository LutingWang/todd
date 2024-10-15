# pylint: disable=duplicate-code

import torch
import torchvision.transforms.v2 as tf_v2
from einops.layers.torch import Rearrange

import todd.tasks.natural_language_processing as nlp
from todd.datasets import CLIP_MEAN, CLIP_STD, coco_url
from todd.models.modules import CLIPText, CLIPViT
from todd.utils import get_image

url = coco_url('val', 2017, 39769)  # pylint: disable=invalid-name
image = get_image(url)

transforms = tf_v2.Compose([
    torch.from_numpy,
    Rearrange('h w c -> 1 c h w'),
    tf_v2.Resize(256, interpolation=tf_v2.InterpolationMode.BICUBIC),
    tf_v2.CenterCrop(256),
    tf_v2.ToDtype(torch.float32, True),
    tf_v2.Normalize(CLIP_MEAN, CLIP_STD),
])
tensor = transforms(image)

clip_vit_b_32 = CLIPViT(
    patch_size=32,
    patch_wh=(7, 7),
    out_features=512,
)
clip_vit_b_32.load_pretrained('pretrained/clip/ViT-B-32.pt')
clip_vit_b_32.requires_grad_(False)
clip_vit_b_32.eval()
cls_, x = clip_vit_b_32(tensor, False)
assert torch.allclose(
    cls_[:, :3],
    torch.tensor([[-0.0084134, 0.0057878, -0.021684]]),
    atol=1e-6,
)
assert torch.allclose(
    x[:, :3, :3],
    torch.tensor([[
        [-0.0022, -0.0041, -0.0005],
        [-0.0010, 0.0007, -0.0157],
        [0.0072, -0.0037, 0.0046],
    ]]),
    atol=1e-4,
)

clip_vit_l_14 = CLIPViT(
    patch_size=14,
    patch_wh=(16, 16),
    width=1024,
    depth=24,
    num_heads=16,
    out_features=768,
)
clip_vit_l_14.load_pretrained('pretrained/clip/ViT-L-14.pt')
clip_vit_l_14.requires_grad_(False)
clip_vit_l_14.eval()
cls_, x = clip_vit_l_14(tensor, False)
assert torch.allclose(
    cls_[:, :3],
    torch.tensor([[-0.037792, 0.072532, 0.008264]]),
    atol=1e-6,
)
assert torch.allclose(
    x[:, :3, :3],
    torch.tensor([[
        [-0.0255, -0.0113, -0.0196],
        [-0.0146, 0.0160, 0.0023],
        [0.0206, 0.0198, 0.0035],
    ]]),
    atol=1e-4,
)

tokenizer = nlp.tokenizers.CLIPTokenizer()
texts = ['hello, world']
# tokens = clip.adaptively_tokenize(texts)
tokens = tokenizer.encodes(texts)

model = CLIPText(out_features=512)
model.load_pretrained('pretrained/clip/ViT-B-32.pt')
model.requires_grad_(False)
model.eval()

x = model(tokens)
eos = CLIPText.eos(tokens, x)
assert torch.allclose(
    eos[:, :3],
    torch.tensor([[7.2253e-02, 1.0570e-01, -1.0538e-01]]),
    atol=1e-5,
)
