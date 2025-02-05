import argparse
import pathlib
from typing import Literal, TextIO, TypedDict

import torch
from bs4 import BeautifulSoup, NavigableString
from ebooklib import ITEM_DOCUMENT, epub
from tqdm import tqdm
from transformers import DynamicCache, Qwen2ForCausalLM, Qwen2TokenizerFast

import todd


class Message(TypedDict):
    role: Literal['system', 'user', 'assistant']
    content: str


class Chatbot:
    PRETRAINED = 'pretrained/qwen/Qwen2.5-7B-Instruct'

    def __init__(self) -> None:
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.PRETRAINED)
        self._tokenizer: Qwen2TokenizerFast = tokenizer

        model = Qwen2ForCausalLM.from_pretrained(
            self.PRETRAINED,
            device_map='auto',
            torch_dtype='auto',
        )
        self._model: Qwen2ForCausalLM = model

        self._refresh()

    def _refresh(self) -> None:
        # todd.logger.debug("Refreshing.")

        self._cache = DynamicCache()

        message = Message(
            role='system',
            content=(
                "你是一个出色的翻译，正在翻译萨尔曼可汗（Salman Khan）编写的《教育新语》（Brave New Words）。"
            ),
        )
        self._conversation = [message]

    def __call__(self, text: str) -> str:
        message = Message(role='user', content=text)
        self._conversation.append(message)

        inputs = self._tokenizer.apply_chat_template(
            self._conversation,
            add_generation_prompt=True,
            return_tensors='pt',
            return_dict=True,
        )
        if todd.Store.cuda:  # pylint: disable=using-constant-test
            inputs = inputs.to('cuda')

        input_ids: torch.Tensor = inputs['input_ids']
        _, input_length = input_ids.shape

        output_ids = self._model.generate(
            **inputs,
            past_key_values=self._cache,
            use_cache=True,
            max_new_tokens=1024,
            top_p=0.95,
        )

        while self._cache.get_seq_length() > 32_000:
            self._refresh()
            self._conversation.append(message)

        generated_ids = output_ids[0, input_length:]
        generated_text = self._tokenizer.decode(generated_ids, True)

        message = Message(role='assistant', content=generated_text)
        self._conversation.append(message)

        return generated_text


class Translator:
    PROMPT = "逐字地把下面的英文文本翻译成中文，不要输出不相关或不符合原文的内容：\n"

    def __init__(self, f: TextIO) -> None:
        self._f = f
        self._chatbot = Chatbot()

    def _translate_text(self, text: str) -> str | None:
        if sum(c.isalpha() for c in text) <= 1:
            return None
        return self._chatbot(self.PROMPT + text)

    def _translate_item(self, item: epub.EpubItem) -> None:
        soup = BeautifulSoup(item.content, 'html.parser')
        body = soup.body
        assert body is not None
        texts: list[NavigableString] = body.find_all(string=True)
        for text in tqdm(texts, leave=False):
            translation = self._translate_text(text)
            if translation is not None and translation.strip():
                # todd.logger.debug("\n'%s' -> '%s'", text, translation)
                self._f.write(f"'{text}' -> '{translation}'\n")
                text.replace_with(translation)
        item.set_content(soup.encode())
        self._f.flush()

    def _translate_book(self, book: epub.EpubBook) -> None:
        items: list[epub.EpubItem] = book.items
        for item in tqdm(items):
            if item.get_type() == ITEM_DOCUMENT:
                self._translate_item(item)

    def translate(self, book: epub.EpubBook) -> None:
        self._translate_book(book)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=pathlib.Path)
    parser.add_argument('output_path', type=pathlib.Path)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    book = epub.read_epub(args.input_path)

    with open('tmp.log', 'w') as f:
        translator = Translator(f)
        translator.translate(book)

    epub.write_epub(args.output_path, book)


if __name__ == '__main__':
    main()
