from argparse import ArgumentParser
from data import *


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--gen-h5', action='store_true',
                        help='Generate images.h5')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.gen_h5:
        gen_images_h5_file()
        return


if __name__ == '__main__':
    main()