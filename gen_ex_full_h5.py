from data import *
import h5py


def gen_ex_images_h5_file2(ex_ids):
    if ex_images_h5_file.exists():
        print(f'Found: {ex_images_h5_file}')
        return

    with h5py.File(str(ex_images_h5_file), 'a') as fp:
        for image_id in progress_bar(ex_ids):
            try:
                im = load_4ch_image_ex(image_id)
            except:
                continue

            im = np.array(im)

            key = 'ex/{}'.format(str(image_id))
            print(key)
            fp.create_dataset(key, data=im)


def main():
    ex_ids = get_ex_ids_to_save(full=True)
    gen_ex_images_h5_file2(ex_ids)


if __name__ == '__main__':
    main()
    
