import pathlib
from typing import TypedDict

import torch
import torch.distributed as dist
import torchvision.transforms.v2 as tf_v2
from tqdm import tqdm

import todd
from todd.datasets import CLIP_MEAN, CLIP_STD, PILDataset
from todd.datasets.access_layers import PILAccessLayer
from todd.models.modules import DINO, CLIPViT
from todd.patches.torch import PrefetchDataLoader, get_rank, get_world_size

DATA_ROOT = pathlib.Path('data/imagenet-21k')
WORK_DIR = pathlib.Path('work_dirs/imagenet-21k')

WORK_DIR.mkdir(parents=True, exist_ok=True)


class T(TypedDict):
    id_: str
    image: torch.Tensor


Batch = tuple[list[str], torch.Tensor]


class Dataset(PILDataset[T]):

    def __init__(self, *args, category: str, **kwargs) -> None:
        access_layer = PILAccessLayer(str(DATA_ROOT), category, suffix='JPEG')
        super().__init__(*args, access_layer=access_layer, **kwargs)

    def __getitem__(self, index: int) -> T:
        key, image = self._access(index)
        tensor = self._transform(image)
        return T(id_=key, image=tensor)


def check(out_path: pathlib.Path) -> bool:
    if not out_path.exists():
        return False
    try:
        torch.load(out_path)
    except Exception:  # pylint: disable=broad-exception-caught
        todd.logger.warning("Failed to load %s. Overwriting...", out_path)
        return False
    todd.logger.debug("%s already exists.", out_path)
    return True


def main() -> None:
    dist.init_process_group('nccl')
    torch.cuda.set_device(get_rank() % torch.cuda.device_count())

    is_master = get_rank() == 0

    # FIXME: 256 should be 224
    transforms = tf_v2.Compose([
        tf_v2.Resize(256, tf_v2.InterpolationMode.BICUBIC),
        tf_v2.CenterCrop(256),
        tf_v2.ToDtype(torch.float32, True),
        tf_v2.Normalize(CLIP_MEAN, CLIP_STD),
    ])
    transforms = transforms.cuda()

    def collate_fn(batch: list[T]) -> Batch:
        ids = [item['id_'] for item in batch]
        images = torch.stack([transforms(item['image']) for item in batch])
        return ids, images

    clip_vit_l_14 = CLIPViT(
        patch_size=14,
        patch_wh=(16, 16),
        width=1024,
        depth=24,
        num_heads=16,
        out_features=768,
    )
    clip_vit_l_14 = clip_vit_l_14.cuda()

    clip_vit_l_14.load_pretrained('pretrained/clip/ViT-L-14.pt')
    clip_vit_l_14.requires_grad_(False)
    clip_vit_l_14.eval()

    dino = DINO()
    dino = dino.cuda()

    dino.load_pretrained('pretrained/dino/dino_vitbase16_pretrain.pth')
    dino.requires_grad_(False)
    dino.eval()

    categories = sorted(category.name for category in DATA_ROOT.iterdir())
    # categories = todd.patches.py_.json_load('wnids.json')
    categories = categories[get_rank()::get_world_size()]

    for category in tqdm(categories, disable=not is_master):
        out_path = WORK_DIR / f'{category}.pth'
        if check(out_path):
            continue

        dataset = Dataset(category=category)
        dataloader: PrefetchDataLoader[Batch] = PrefetchDataLoader(
            dataset,
            16,
            num_workers=2,
            collate_fn=collate_fn,
        )

        id_list: list[str] = []
        clip_feature_list: list[torch.Tensor] = []
        dino_feature_list: list[torch.Tensor] = []
        for ids, images in dataloader:
            clip_features, _ = clip_vit_l_14(images, False)
            dino_features, _ = dino(images, False)

            id_list.extend(ids)
            clip_feature_list.append(clip_features)
            dino_feature_list.append(dino_features)

        data = dict(
            ids=id_list,
            clip_features=torch.cat(clip_feature_list).half(),
            dino_features=torch.cat(dino_feature_list).half(),
        )
        torch.save(data, out_path)


if __name__ == '__main__':
    main()
