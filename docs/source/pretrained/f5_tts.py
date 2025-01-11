import argparse
import pathlib

import torch

import todd.tasks.text_to_speech as tts
from todd.utils import init_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('voices', type=pathlib.Path)
    parser.add_argument('lines', type=pathlib.Path)
    parser.add_argument('output', type=pathlib.Path)
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main() -> None:
    args = parse_args()

    init_seed(args.seed)

    f5_tts = tts.f5_tts.F5_TTS(args.voices)
    f5_tts.run(args.lines, output_file=args.output)


if __name__ == '__main__':
    main()
