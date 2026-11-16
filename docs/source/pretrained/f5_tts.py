import argparse
from pathlib import Path

import torch

import todd_tasks.text_to_speech as tts
from todd_torch.utils import init_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('voices_file', type=Path)
    parser.add_argument('lines_file', type=Path)
    parser.add_argument('output_file', type=Path)
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main() -> None:
    args = parse_args()

    init_seed(args.seed)

    f5_tts = tts.f5_tts.F5_TTS(args.voices_file)
    f5_tts.run(args.lines_file, output_file=args.output_file)


if __name__ == '__main__':
    main()
