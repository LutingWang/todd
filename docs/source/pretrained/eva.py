from unittest.mock import patch
import torch
import torchvision.transforms.v2 as tf_v2
from einops.layers.torch import Rearrange
from PIL import Image

import todd.tasks.natural_language_processing as nlp
from todd.datasets import CLIP_MEAN, CLIP_STD
from todd.models.modules import EVA_CLIPText, EVA_CLIPViT


def preprocess(image_size: int) -> tf_v2.Compose:
    transforms = tf_v2.Compose([
        tf_v2.PILToTensor(),
        Rearrange('c h w -> 1 c h w'),
        tf_v2.Resize(
            image_size,
            interpolation=tf_v2.InterpolationMode.BICUBIC,
        ),
        tf_v2.CenterCrop(image_size),
        tf_v2.ToDtype(torch.float32, True),
        tf_v2.Normalize(CLIP_MEAN, CLIP_STD),
    ])
    with Image.open('CLIP.png') as image:
        return transforms(image.convert('RGB'))


def forward(
    pretrained: str,
    vit: EVA_CLIPViT,
    text: EVA_CLIPText,
    image: torch.Tensor,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    vit = vit.cuda()
    vit.load_pretrained(pretrained)
    vit.requires_grad_(False)
    vit.eval()
    vit_cls, vit_x = vit(image, False)

    text = text.cuda()
    text.load_pretrained(pretrained)
    text.requires_grad_(False)
    text.eval()
    text_x = text(tokens)
    text_eos = EVA_CLIPText.eos(tokens, text_x)

    logits = torch.einsum('v d, t d -> v t', vit_cls, text_eos)
    logits = logits * 100
    p = logits.softmax(-1)

    return vit_cls, vit_x, text_eos, p


def main() -> None:
    image_224 = preprocess(224)
    image_224 = image_224.cuda()

    image_336 = preprocess(336)
    image_336 = image_336.cuda()

    tokenizer = nlp.tokenizers.CLIPTokenizer()
    texts = ['a diagram', 'a dog', 'a cat']
    tokens = tokenizer.encodes(texts)
    tokens = tokens.cuda()

    # EVA01-CLIP-g-14
    cls_, x, eos, p = forward(
        'pretrained/eva/eva_clip/EVA01_CLIP_g_14_psz14_s11B.pt',
        EVA_CLIPViT(
            depth=40,
            width=1408,
            patch_size=14,
            patch_wh=(16, 16),
            block_kwargs=dict(
                num_heads=16,
                mlp_ratio=4.3637,
            ),
            out_features=1024,
        ),
        EVA_CLIPText(
            width=768,
            block_kwargs=dict(num_heads=12),
            out_features=1024,
        ),
        image_224,
        tokens,
    )
    print(p)

    # EVA01-CLIP-g-14-plus
    cls_, x, eos, p = forward(
        'pretrained/eva/eva_clip/EVA01_CLIP_g_14_plus_psz14_s11B.pt',
        EVA_CLIPViT(
            depth=40,
            width=1408,
            patch_size=14,
            patch_wh=(16, 16),
            block_kwargs=dict(
                num_heads=16,
                mlp_ratio=4.3637,
            ),
            out_features=1024,
        ),
        EVA_CLIPText(
            width=1024,
            depth=24,
            block_kwargs=dict(num_heads=16),
            out_features=1024,
        ),
        image_224,
        tokens,
    )
    print(p)

    # EVA02-CLIP-B-16
    cls_, x, eos, p = forward(
        'pretrained/eva/eva_clip/EVA02_CLIP_B_psz16_s8B.pt',
        EVA_CLIPViT(
            block_kwargs=dict(
                mlp_ratio=2.6667,
                attention_with_norm=True,
                mlp_type='swiglu',
                rope=True,
            ),
            out_features=512,
        ),
        EVA_CLIPText(block_kwargs=dict(num_heads=8), out_features=512),
        image_224,
        tokens,
    )
    print(p)

    # EVA02-CLIP-L-14
    cls_, x, eos, p = forward(
        'pretrained/eva/eva_clip/EVA02_CLIP_L_psz14_s4B.pt',
        EVA_CLIPViT(
            depth=24,
            width=1024,
            patch_size=14,
            patch_wh=(16, 16),
            block_kwargs=dict(
                num_heads=16,
                mlp_ratio=2.6667,
                attention_with_norm=True,
                mlp_type='swiglu',
                rope=True,
            ),
            out_features=768,
        ),
        EVA_CLIPText(
            width=768,
            block_kwargs=dict(num_heads=12),
            out_features=768,
        ),
        image_224,
        tokens,
    )
    print(p)

    # EVA02-CLIP-L-14-336
    cls_, x, eos, p = forward(
        'pretrained/eva/eva_clip/EVA02_CLIP_L_psz14_224to336.pt',
        EVA_CLIPViT(
            depth=24,
            width=1024,
            patch_size=14,
            patch_wh=(24, 24),
            block_kwargs=dict(
                num_heads=16,
                mlp_ratio=2.6667,
                attention_with_norm=True,
                mlp_type='swiglu',
                rope=True,
            ),
            out_features=768,
        ),
        EVA_CLIPText(
            width=768,
            block_kwargs=dict(num_heads=12),
            out_features=768,
        ),
        image_336,
        tokens,
    )
    print(p)

    # EVA02-CLIP-bigE-14
    cls_, x, eos, p = forward(
        'pretrained/eva/eva_clip/EVA02_CLIP_E_psz14_s4B.pt',
        EVA_CLIPViT(
            depth=64,
            width=1792,
            patch_size=14,
            patch_wh=(16, 16),
            block_kwargs=dict(
                num_heads=16,
                mlp_ratio=8.571428571428571,
                post_norm=True,
            ),
            out_features=1024,
        ),
        EVA_CLIPText(
            width=1024,
            depth=24,
            block_kwargs=dict(num_heads=16),
            out_features=1024,
        ),
        image_224,
        tokens,
    )
    print(p)

    # EVA02-CLIP-bigE-14-plus
    cls_, x, eos, p = forward(
        'pretrained/eva/eva_clip/EVA02_CLIP_E_psz14_plus_s9B.pt',
        EVA_CLIPViT(
            depth=64,
            width=1792,
            patch_size=14,
            patch_wh=(16, 16),
            block_kwargs=dict(
                num_heads=16,
                mlp_ratio=8.571428571428571,
                post_norm=True,
            ),
            out_features=1024,
        ),
        EVA_CLIPText(
            width=1280,
            depth=32,
            block_kwargs=dict(num_heads=20),
            out_features=1024,
        ),
        image_224,
        tokens,
    )
    print(p)


if __name__ == '__main__':
    main()
