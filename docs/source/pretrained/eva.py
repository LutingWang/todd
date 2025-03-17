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

    vit_cls: torch.Tensor
    vit_x: torch.Tensor
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

    return vit_cls.cpu(), vit_x.cpu(), text_eos.cpu(), p.cpu()


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
    assert torch.allclose(
        cls_[:, :3],
        torch.tensor([[-0.0245, -0.0377, 0.0734]]),
        atol=1e-4,
    )
    assert torch.allclose(
        x[:, :3, :3],
        torch.tensor([[
            [-0.0403, -0.0141, 0.0704],
            [-0.0194, -0.0311, 0.0641],
            [-0.0143, -0.0324, 0.0766],
        ]]),
        atol=1e-4,
    )
    assert torch.allclose(
        eos[:, :3],
        torch.tensor([
            [-0.0464, -0.0281, 0.0053],
            [-0.0092, -0.0078, -0.0510],
            [-0.0177, -0.0270, -0.0450],
        ]),
        atol=1e-4,
    )
    assert torch.allclose(
        p,
        torch.tensor([[1., 0., 0.]]),
        atol=1e-4,
    )

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
    assert torch.allclose(
        cls_[:, :3],
        torch.tensor([[-0.0368, -0.0143, 0.0289]]),
        atol=1e-4,
    )
    assert torch.allclose(
        x[:, :3, :3],
        torch.tensor([[
            [-0.0308, -0.0182, 0.0376],
            [-0.0325, -0.0316, 0.0314],
            [-0.0227, -0.0452, 0.0487],
        ]]),
        atol=1e-4,
    )
    assert torch.allclose(
        eos[:, :3],
        torch.tensor([
            [-0.0110, -0.0253, -0.0207],
            [-0.0149, 0.0108, 0.0053],
            [-0.0274, 0.0049, -0.0001],
        ]),
        atol=1e-4,
    )
    assert torch.allclose(
        p,
        torch.tensor([[0.9679, 0.0183, 0.0138]]),
        atol=1e-4,
    )

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
    assert torch.allclose(
        cls_[:, :3],
        torch.tensor([[0.0037, 0.0008, -0.0523]]),
        atol=1e-4,
    )
    assert torch.allclose(
        x[:, :3, :3],
        torch.tensor([[
            [-0.0034, 0.0186, 0.0485],
            [-0.0409, -0.0164, -0.0206],
            [-0.0445, 0.0055, -0.0461],
        ]]),
        atol=1e-4,
    )
    assert torch.allclose(
        eos[:, :3],
        torch.tensor([
            [0.0128, -0.0275, -0.0009],
            [0.0308, 0.0111, -0.0259],
            [0.0563, 0.0133, -0.0017],
        ]),
        atol=1e-4,
    )
    assert torch.allclose(
        p,
        torch.tensor([[0.8906, 0.0806, 0.0288]]),
        atol=1e-4,
    )

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
    assert torch.allclose(
        cls_[:, :3],
        torch.tensor([[0.0084, -0.0344, 0.0447]]),
        atol=1e-4,
    )
    assert torch.allclose(
        x[:, :3, :3],
        torch.tensor([[
            [-0.0555, 0.0149, 0.0721],
            [0.0397, -0.0015, -0.0031],
            [0.1327, 0.0320, 0.0839],
        ]]),
        atol=1e-4,
    )
    assert torch.allclose(
        eos[:, :3],
        torch.tensor([
            [0.0618, 0.0079, 0.0515],
            [0.0667, 0.0064, 0.0298],
            [0.0640, 0.0068, 0.0372],
        ]),
        atol=1e-4,
    )
    assert torch.allclose(
        p,
        torch.tensor([[0.9870, 0.0113, 0.0017]]),
        atol=1e-4,
    )

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
    assert torch.allclose(
        cls_[:, :3],
        torch.tensor([[0.0015, -0.0080, 0.0175]]),
        atol=1e-4,
    )
    assert torch.allclose(
        x[:, :3, :3],
        torch.tensor([[
            [-0.0625, -0.0046, 0.0795],
            [-0.0190, 0.0481, 0.0284],
            [0.0500, 0.0215, -0.0026],
        ]]),
        atol=1e-4,
    )
    assert torch.allclose(
        eos[:, :3],
        torch.tensor([
            [0.0638, 0.0100, 0.0487],
            [0.0685, 0.0098, 0.0287],
            [0.0657, 0.0102, 0.0357],
        ]),
        atol=1e-4,
    )
    assert torch.allclose(
        p,
        torch.tensor([[9.9201e-01, 7.1911e-03, 8.0148e-04]]),
        atol=1e-5,
    )

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
    assert torch.allclose(
        cls_[:, :3],
        torch.tensor([[0.0048, -0.0075, 0.0696]]),
        atol=1e-4,
    )
    assert torch.allclose(
        x[:, :3, :3],
        torch.tensor([[
            [-0.0526, -0.0071, 0.0447],
            [-0.0309, -0.0390, 0.0307],
            [0.0067, -0.0041, 0.0520],
        ]]),
        atol=1e-4,
    )
    assert torch.allclose(
        eos[:, :3],
        torch.tensor([
            [0.0065, -0.0269, -0.0197],
            [0.0299, -0.0469, -0.0125],
            [0.0420, -0.0402, -0.0193],
        ]),
        atol=1e-4,
    )
    assert torch.allclose(
        p,
        torch.tensor([[9.9761e-01, 1.9345e-03, 4.5057e-04]]),
        atol=1e-5,
    )

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
    assert torch.allclose(
        cls_[:, :3],
        torch.tensor([[0.0208, 0.0306, -0.0118]]),
        atol=1e-4,
    )
    assert torch.allclose(
        x[:, :3, :3],
        torch.tensor([[
            [0.0352, -0.0073, 0.0306],
            [-0.0203, 0.0365, 0.0168],
            [0.0326, 0.0032, 0.0037],
        ]]),
        atol=1e-4,
    )
    assert torch.allclose(
        eos[:, :3],
        torch.tensor([
            [-0.0548, 0.0343, 0.0294],
            [-0.0568, 0.0284, 0.0246],
            [-0.0420, -0.0061, 0.0130],
        ]),
        atol=1e-4,
    )
    assert torch.allclose(
        p,
        torch.tensor([[0.9871, 0.0105, 0.0024]]),
        atol=1e-4,
    )


if __name__ == '__main__':
    main()
