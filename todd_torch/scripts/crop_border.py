#!python3

import argparse

import cv2
import numpy as np

# TODO: cv2.CropBorder or something


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Crop border")
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('-t', '--threshold', type=int, default=250)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    image = cv2.imread(args.input_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray <= args.threshold
    columns = np.any(mask, axis=0)
    rows = np.any(mask, axis=1)
    x1, x2 = np.where(rows)[0][[0, -1]]
    y1, y2 = np.where(columns)[0][[0, -1]]
    image = image[x1:x2 + 1, y1:y2 + 1]
    cv2.imwrite(args.output_path, image)


if __name__ == '__main__':
    main()
