# pylint: disable=duplicate-code

import pathlib
from typing import Sequence

import torch
from PIL import Image
from transformers import (
    BatchEncoding,
    BatchFeature,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    Qwen2TokenizerFast,
)

import todd


class Chatbot:
    PRETRAINED = 'pretrained/qwen/Qwen2.5-VL-7B-Instruct'

    def __init__(self) -> None:
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.PRETRAINED)
        self._tokenizer: Qwen2TokenizerFast = tokenizer

        processor = Qwen2_5_VLProcessor.from_pretrained(
            self.PRETRAINED,
            max_pixels=1024 * 28 * 28,
        )
        self._processor = processor

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.PRETRAINED,
            device_map='auto',
            torch_dtype='auto',
        )
        self._model = model

    def __call__(
        self,
        inputs: BatchEncoding | BatchFeature,
    ) -> tuple[torch.Tensor, str]:
        if todd.Store.cuda:  # pylint: disable=using-constant-test
            inputs = inputs.to('cuda')

        input_ids: torch.Tensor = inputs.input_ids
        _, input_length = input_ids.shape

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=1024,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

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
        generated_ids = output_ids[0, input_length:]
        generated_text = self._processor.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return hidden_states, generated_text

    def chat(self, text: str) -> tuple[torch.Tensor, str]:
        conversation = [dict(role='user', content=text)]
        inputs = self._tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors='pt',
            return_dict=True,
        )
        return self(inputs)

    def chat_multimodal(
        self,
        images: Sequence[Image.Image],
        text: str,
    ) -> tuple[torch.Tensor, str]:
        content = [dict(type='image') for _ in images]
        content.append(dict(type='text', text=text))
        conversation = [dict(role='user', content=content)]
        input_text = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )

        inputs = self._processor(
            images,
            input_text,
            add_special_tokens=False,
            return_tensors='pt',
        )
        return self(inputs)


def main() -> None:
    assert torch.cuda.device_count() <= 4, (
        "Please use no more than 4 GPUs, in order to avoid RuntimeError."
    )

    images_root = pathlib.Path(__file__).parent / 'images'
    images = [Image.open(image) for image in images_root.iterdir()]

    chatbot = Chatbot()

    hidden_states, response = chatbot.chat("What is AI?")
    todd.logger.info(hidden_states)
    todd.logger.info(response)

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
