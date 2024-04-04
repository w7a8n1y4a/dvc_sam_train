import glob
import random
import yaml
import shutil

import checksumdir

from core.config import settings
from core.utils import dict_to_log, create_dirs, dict_to_json_file_save
from mask_gen import gen_dataset


def split(params, images_path, masks_path) -> int:
    images_path_list = glob.glob(images_path + '/*')
    masks_path_list = glob.glob(masks_path + '/*')

    images_path_list.sort()
    masks_path_list.sort()

    img_mask_list = [item for item in zip(images_path_list, masks_path_list)]

    if params['shuffle_dataset']:
        random.shuffle(img_mask_list)

    train_list = img_mask_list[: int(len(img_mask_list) * params['train_size'] / 100)]
    test_list = img_mask_list[-int(len(img_mask_list) * params['test_size'] / 100) :]

    create_dirs(
        [
            f'{settings.app_path}/split/{dataset}/{split}/'
            for split in ['images', 'masks']
            for dataset in ['train', 'test']
        ]
    )

    for img, mask in train_list:
        shutil.move(img, f'{settings.app_path}/split/train/images/{img.split("/")[-1]}')
        shutil.move(mask, f'{settings.app_path}/split/train/masks/{mask.split("/")[-1]}')

    for img, mask in test_list:
        shutil.move(img, f'{settings.app_path}/split/test/images/{img.split("/")[-1]}')
        shutil.move(mask, f'{settings.app_path}/split/test/masks/{mask.split("/")[-1]}')

    dict_to_log({'train_size': len(train_list), 'test_size': len(test_list)})

    return len(img_mask_list)


def main() -> None:
    # параметры сплита
    params = yaml.safe_load(open("params.yaml"))["split"]

    # удаляет старый сплит, в dvc можно так делать, потому что хэши будут сверены в процессе
    try:
        shutil.rmtree(f'{settings.app_path}/split')
    except FileNotFoundError:
        dict_to_log({'info': "No alter split"})

    images_path = f'{settings.app_path}/tmp/images'
    masks_path = f'{settings.app_path}/tmp/masks'

    # генерирует датасет
    md5_images = gen_dataset(images_path, masks_path)

    # разделяет датасет и складывает его по директориям
    counter = split(params, images_path, masks_path)

    shutil.rmtree('tmp/images')
    shutil.rmtree('tmp/masks')

    dataset_yaml_dict = {
        'data_hashes': [checksumdir.dirhash(f'{settings.app_path}/split/{item}', 'md5') for item in ['train', 'test']]
    }

    dict_to_json_file_save(dataset_yaml_dict, './split/dataset_data.json')

    dict_to_log(dataset_yaml_dict)


if __name__ == "__main__":
    main()
