import os
import random

import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    MllamaForConditionalGeneration,
    MllamaProcessor,
    PreTrainedTokenizerFast,
)

import todd


class Chatbot:
    PRETRAINED = 'pretrained/llama/Llama-3.2-11B-Vision-Instruct'

    def __init__(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.PRETRAINED)
        self._tokenizer: PreTrainedTokenizerFast = tokenizer

        processor = MllamaProcessor.from_pretrained(self.PRETRAINED)
        self._processor = processor

        model = MllamaForConditionalGeneration.from_pretrained(
            self.PRETRAINED,
            device_map='auto',
            torch_dtype='auto',
        )
        self._model = model

    def __call__(self, inputs: BatchEncoding) -> str:
        if todd.Store.cuda:  # pylint: disable=using-constant-test
            inputs = inputs.to('cuda')

        input_ids: torch.Tensor = inputs['input_ids']
        _, input_length = input_ids.shape

        output_ids = self._model.generate(**inputs, max_new_tokens=1024)
        generated_ids = output_ids[0, input_length:]
        generated_text = self._processor.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return generated_text

    def chat(self, text: str) -> str:
        conversation = [dict(role='user', content=text)]
        inputs = self._tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors='pt',
            return_dict=True,
        )
        return self(inputs)

    def chat_multimodal(self, images: list[Image.Image], text: str) -> str:
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
    filenames = os.listdir('cat')
    sampled_filenames = random.sample(filenames, 5)
    images = [
        Image.open(os.path.join('cat', filename))
        for filename in sampled_filenames
    ]

    chatbot = Chatbot()

    response = chatbot.chat("What is AI?")
    todd.logger.info(response)

    caption = chatbot.chat_multimodal(
        images,
        "The images are exemplars of a category. "
        "Can you guess what category it is? "
        "Answer with a template of the form: A photo of <object>. "
        "Example: A photo of cat.",
    )
    todd.logger.info(caption)


if __name__ == '__main__':
    main()
