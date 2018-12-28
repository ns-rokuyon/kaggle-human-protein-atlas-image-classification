from data import *
from argparse import ArgumentParser


N = 4


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--index', required=True, type=int)
    return parser.parse_args()


def main():
    setup()

    args = parse_args()
    index = args.index

    ex_ids = get_ex_ids_to_save(full=True)
    split_ex_ids = np.array_split(ex_ids, N)
    print(f'Index: {index}')

    save_ex_images(index, list(split_ex_ids[index]))


if __name__ == '__main__':
    main()