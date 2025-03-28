import pathlib
from typing import Sequence

import einops
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch import nn
from transformers import AutoModel, Qwen2Tokenizer
from transformers.generation import GenerateDecoderOnlyOutput

import todd
from todd.datasets import IMAGENET_MEAN, IMAGENET_STD

PRETRAINED = 'pretrained/intern/InternVL2_5-1B'


class PILToTensor(transforms.PILToTensor):

    def __init__(self, *args, patch_wh: tuple[int, int], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._patch_wh = patch_wh

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert('RGB')

        image_w, image_h = image.size
        patch_w, patch_h = self._patch_wh

        nw = round(image_w / patch_w)
        nh = round(image_h / patch_h)

        patches = image.resize(
            (nw * patch_w, nh * patch_h),
            Image.Resampling.BILINEAR,
        )
        thumbnail = image.resize(self._patch_wh, Image.Resampling.BILINEAR)

        tensors = (
            einops.rearrange(
                super().__call__(patches),
                'c (nh patch_h) (nw patch_w) -> (nh nw) c patch_h patch_w',
                nh=nh,
                patch_h=patch_h,
                nw=nw,
                patch_w=patch_w,
            ),
            einops.rearrange(
                super().__call__(thumbnail),
                'c patch_h patch_w -> 1 c patch_h patch_w',
            ),
        )
        return torch.cat(tensors)


def get_device_map(depth: int) -> dict[str, int]:
    device_map: dict[str, int] = dict()
    device_map = {
        'vision_model': 0,
        'mlp1': 0,
        'language_model.model.embed_tokens': 0,
        'language_model.model.rotary_emb': 0,
        'language_model.model.norm': 0,
        'language_model.lm_head': 0,
    }

    world_size = torch.cuda.device_count()
    if world_size <= 1:
        device_map['language_model.model.layers'] = 0
        return device_map

    # Distribute layers evenly across GPUs except GPU 0
    device_map.update({
        f'language_model.model.layers.{i}': j + 1
        for i, j in enumerate(
            sorted((k * (world_size - 1)) // depth for k in range(depth)),
        )
    })

    return device_map


class Model:
    IMAGE_TOKEN = '<IMG_CONTEXT>'  # nosec: B105
    EOS_TOKEN = '<|im_end|>'  # nosec: B105
    IMAGE_START_TOKEN = '<img>'  # nosec: B105
    IMAGE_END_TOKEN = '</img>'  # nosec: B105
    IMAGE_WH = (448, 448)
    DEPTH = 24

    def __init__(self, *, image_token_id: int, **kwargs) -> None:
        model = AutoModel.from_pretrained(
            PRETRAINED,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map=get_device_map(self.DEPTH),
        )
        model.img_context_token_id = image_token_id
        self._model = model

        self._kwargs = kwargs

    @property
    def num_tokens_per_image(self) -> int:
        return self._model.num_image_token

    @property
    def first_parameter(self) -> nn.Parameter:
        return next(self._model.parameters())

    def apply_template(self, text: str) -> str:
        template = self._model.conv_template.copy()
        template.append_message(template.roles[0], text)
        template.append_message(template.roles[1], None)
        return template.get_prompt()

    def generate(self, *args, **kwargs) -> GenerateDecoderOnlyOutput:
        return self._model.generate(*args, **kwargs, **self._kwargs)


class Chatbot:
    MODEL_TYPE = Model

    def __init__(self) -> None:
        tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(
            PRETRAINED,
            trust_remote_code=True,
            use_fast=False,
        )
        self._tokenizer = tokenizer

        self._transform = transforms.Compose([
            PILToTensor(patch_wh=self.MODEL_TYPE.IMAGE_WH),
            transforms.ToDtype(torch.float32, True),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        self._model = self.MODEL_TYPE(
            image_token_id=tokenizer.convert_tokens_to_ids(
                self.MODEL_TYPE.IMAGE_TOKEN,
            ),
            eos_token_id=tokenizer.convert_tokens_to_ids(
                self.MODEL_TYPE.EOS_TOKEN,
            ),
            max_new_tokens=256,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

    def _preprocess_image(
        self,
        i: int,
        image: Image.Image,
    ) -> tuple[torch.Tensor, str]:
        tensor: torch.Tensor = self._transform(image)
        text = (
            f'Image-{i}: ' + self._model.IMAGE_START_TOKEN
            + self._model.IMAGE_TOKEN * self._model.num_tokens_per_image
            * tensor.shape[0] + self._model.IMAGE_END_TOKEN
        )
        return tensor, text

    def chat_multimodal(
        self,
        images: Sequence[Image.Image],
        text: str,
    ) -> tuple[torch.Tensor, str]:
        tensors: list[torch.Tensor] = []
        texts: list[str] = []
        for i, image in enumerate(images):
            tensor_, text_ = self._preprocess_image(i, image)
            tensors.append(tensor_)
            texts.append(text_)
        input_image = torch.cat(tensors).to(self._model.first_parameter)
        input_text = self._model.apply_template('\n'.join(texts + [text]))

        inputs = self._tokenizer(input_text, return_tensors='pt')
        if todd.Store.cuda:  # pylint: disable=using-constant-test
            inputs = inputs.to('cuda')

        outputs = self._model.generate(input_image, **inputs)

        # `hidden_states` is a tuple of `token_hidden_states`.
        # The length of `hidden_states` equals the number of generated tokens.

        # Each `token_hidden_states` is a tuple of `layer_hidden_states`.
        # The length of `token_hidden_states` is one more than the number of
        # layers, since the input embeddings are included.

        # Each `layer_hidden_states` is a tensor of shape
        # batch_size x num_tokens x hidden_size.
        # `num_tokens` in the first `token_hidden_states` equals the length of
        # inputs, while others are one.

        hidden_states = torch.stack([
            hidden_states[-1][:, -1] for hidden_states in outputs.hidden_states
        ])

        output_ids = outputs.sequences
        generated_ids = output_ids[0]
        generated_text = self._tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return hidden_states, generated_text


def main() -> None:
    images_root = pathlib.Path(__file__).parent / 'images'
    images = [Image.open(image) for image in images_root.iterdir()]

    chatbot = Chatbot()

    hidden_states, caption = chatbot.chat_multimodal(
        images,
        "The images are exemplars of a category. "
        "Can you guess what category it is? "
        "Answer with a template of the form: A photo of <object>. "
        "Example: A photo of cat.",
    )
    todd.logger.info(hidden_states)
    todd.logger.info(caption)


if __name__ == '__main__':
    main()
