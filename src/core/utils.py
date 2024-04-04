import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import torch

import os
import time
import json
import logging
import hashlib
import random
from functools import wraps

import git
import requests
from requests import Response

from .config import settings


def show_mask(mask: np.array, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[:2]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def plot_image_mask(image: PIL.Image, mask: PIL.Image, filename: str):
    fig, axes = plt.subplots()
    axes.imshow(np.array(image))
    ground_truth_seg = np.array(mask)
    show_mask(ground_truth_seg, axes)
    axes.title.set_text(f"{filename} predicted mask")
    axes.axis("off")
    plt.savefig("./plots/" + filename + ".jpg")
    plt.close()


def plot_image_mask_dataset(dataset: torch.utils.data.Dataset, idx: int):
    image_path = dataset.img_files[idx]
    mask_path = dataset.mask_files[idx]
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    mask = mask.convert('1')
    plot_image_mask(image, mask)


def get_bounding_box(ground_truth_map: np.array) -> list:
    idx = np.where(ground_truth_map > 0)

    x_indices = idx[1]
    y_indices = idx[0]

    try:
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
    except:
        x_min = 0
        x_max = 0
        y_min = 0
        y_max = 0

    bbox = [x_min, y_min, x_max, y_max]

    return bbox


def stacking_batch(batch, outputs):
    stk_gt = torch.stack([b["ground_truth_mask"] for b in batch], dim=0)
    stk_out = torch.stack([out["low_res_logits"] for out in outputs], dim=0)
    iou = torch.stack([out["iou_predictions"] for out in outputs], dim=0)

    return stk_gt, stk_out, iou


def create_dirs(dirs_paths: list[str]):
    """Функция создания дирректорий
    :param dirs_paths: Пути которые нужно создать
    :return: pass
    """

    for path in dirs_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def filepath_to_md5(filepath: str):
    """Функция хэширования файла по заданному пути
    :param filepath: Путь до хэшируемого файла
    :return: md5 hash
    """

    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def dict_to_json_file_save(dictionary: dict, filepath: str) -> bool:
    """Функция сохранения dict в файл
    :param dictionary: dict
    :param filepath: Путь до файла с сохранённым dict
    :return:
    """

    create_dirs([filepath])

    with open(filepath, "w") as f:
        f.write(str(json.dumps(dictionary, indent=4)))

    return True


def json_file_to_dict(filepath: str) -> dict:
    """Функция преобразования файла в dict
    :param filepath: Путь до файла с сохранённым dict
    :return: dict
    """

    with open(filepath, "r") as f:
        json_file_string = f.read()

    return json.loads(json_file_string)


def dict_to_log(data: dict):
    """Преобразует dict в запись лога
    :param data: dict
    :return: pass and data to log
    """

    logging.warning(json.dumps(data))


def check_mem(torch):
    """Вписывает в лог данные о остатке памяти ГУ
    :param torch: объект torch
    :return: pass and data to log
    """

    allowed, all = torch.cuda.mem_get_info()

    data = {'all_gpu_mem': all / 2**20, 'allowed': allowed / 2**20}

    logging.warning(json.dumps(data))


def generate_random_image():
    """Затычка вместо генерируемых картинок, через matplotlig или любой другой пакет"""

    import numpy
    from PIL import Image

    imarray = numpy.random.rand(512, 512, 3) * 255
    im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
    im.save(f'{os.path.abspath(os.getcwd())}/plot/train/random_image.png')


def exec_time(func):
    """Декоратор проверяющий время выполнения функции"""

    @wraps(func)
    def exec_time_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        dict_to_log({'exec_time': end_time - start_time})
        return result

    return exec_time_wrapper


def split_one_byte_names(
    train_percent: int, valid_percent: int, test_percent: int, randomize: bool = False
) -> tuple[list[str], list[str], list[str]]:
    """Разделяет 1 байт hex чисел в правильной пропорции"""

    assert train_percent + valid_percent + test_percent == 100, 'Not 100'

    alphabet = '0123456789abcdef'
    hex_data_list = [f'{first}{two}' for first in alphabet for two in alphabet]

    if randomize:
        random.shuffle(hex_data_list)

    return (
        hex_data_list[: int(len(hex_data_list) * train_percent / 100)],
        hex_data_list[int(len(hex_data_list) * train_percent / 100) : -int(len(hex_data_list) * test_percent / 100)],
        hex_data_list[-int(len(hex_data_list) * test_percent / 100) :],
    )


def current_branch_name() -> str:
    return git.Repo(search_parent_directories=True).active_branch.name


def send_file_to_telegram(filepath: str) -> Response or None:
    """Отправляет файл в MFR бота
    :param filepath: путь до файла
    :return: Response obj или None
    """

    if settings.MFR_URL and settings.MFR_AUTH_TOKEN:
        with open(filepath, 'rb') as curve:
            response = requests.post(
                settings.MFR_URL, files={'file': curve}, headers={'auth-token': settings.MFR_AUTH_TOKEN}
            )
            dict_to_log({'send_succes': str(response)})

        return response
    return None
