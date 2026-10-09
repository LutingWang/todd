#!python3

import argparse

import cv2


def main() -> None:
    parser = argparse.ArgumentParser("Crop border")
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('-t', '--threshold', type=int, default=250)
    args = parser.parse_args()
    image = cv2.imread(args.input_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, args.threshold, 255, cv2.THRESH_BINARY_INV)
    x, y, width, height = cv2.boundingRect(cv2.findNonZero(mask))
    image = image[y:y + height, x:x + width]
    cv2.imwrite(args.output_path, image)


if __name__ == '__main__':
    main()
