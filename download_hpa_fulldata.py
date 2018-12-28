from data import *


def main():
    setup()

    ex_ids = get_ex_ids_to_save(full=True)
    download_ex_data(ex_ids)


if __name__ == '__main__':
    main()