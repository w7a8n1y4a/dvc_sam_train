import hashlib
import os
import cv2
import numpy
import glob
import uuid
import shutil
import yaml

from multiprocessing import queues, Process, SimpleQueue, cpu_count


from core.utils import dict_to_log

params = yaml.safe_load(open("params.yaml"))["split"]


def split_img_box(uuid: str, img, output_dir: str, width: int, height: int, overlap: float):
    rows = int((img.shape[0] / (height - height * overlap)))
    cols = int((img.shape[1] / (width - width * overlap)))

    for i in range(rows):
        for j in range(cols):
            x = int(j * (width - width * overlap))
            y = int(i * (height - height * overlap))

            crop_img = img[y : y + height, x : x + width]

            if crop_img.shape[0] == height and crop_img.shape[1] == width:
                output_image_path = os.path.join(output_dir, f'{str(uuid)}_{i}_{j}.png')
                cv2.imwrite(output_image_path, crop_img)


def split_img_devider(uuid: str, img, output_dir: str, devider: int, overlap: float):
    devider_overlap = int(devider / overlap)

    height = int(img.shape[0] / devider)
    width = int(img.shape[1] / devider)

    cols = int(img.shape[0] / (img.shape[0] / devider_overlap))
    rows = int(img.shape[1] / (img.shape[1] / devider_overlap))

    for i in range(cols):
        for j in range(rows):
            x = int(j * (width - width * overlap))
            y = int(i * (height - height * overlap))

            crop_img = img[y : y + height, x : x + width]

            if crop_img.shape[0] == height and crop_img.shape[1] == width:
                output_image_path = os.path.join(output_dir, f'{str(uuid)}_{i}_{j}.png')
                cv2.imwrite(output_image_path, crop_img)


def mask_gen(filepath: str) -> str:
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    img = cv2.medianBlur(img, 5)
    ret, th1 = cv2.threshold(img, numpy.median(img), 255, cv2.THRESH_BINARY)

    img_md5 = hashlib.md5(filepath.split('/')[-1].encode()).hexdigest()

    cv2.imwrite(f'img/{img_md5}.png', img)
    cv2.imwrite(f'mask/{img_md5}.png', th1)

    dict_to_log({'start': str(img_md5)})

    split_img_box(img_md5, img, 'tmp/images', params['box_size'], params['box_size'], params['overlap'])
    split_img_box(img_md5, th1, 'tmp/masks', params['box_size'], params['box_size'], params['overlap'])

    dict_to_log({'end': str(img_md5)})

    return str(img_md5)


def worker(jobs, results) -> None:
    while path := jobs.get():
        results.put(mask_gen(path))
    results.put(None)


def start_jobs(procs, jobs, results, images_path):
    for item in images_path:
        jobs.put(item)
    for _ in range(procs):
        proc = Process(target=worker, args=(jobs, results))
        proc.start()
        jobs.put(0)


def get_uuid_images(images_path, results):
    uuid_images = []
    procs_done = 0
    while procs_done < len(images_path):
        uuid = results.get()

        if uuid == None:
            pass
        else:
            procs_done += 1
            uuid_images.append(uuid)

    dict_to_log({'count_processed_images': procs_done})

    return uuid_images


def gen_dataset(path_img, path_mask) -> list[str]:
    try:
        shutil.rmtree(path_img)
    except:
        pass

    try:
        shutil.rmtree(path_mask)
    except:
        pass

    os.makedirs(path_img, exist_ok=True)
    os.makedirs(path_mask, exist_ok=True)

    jobs = SimpleQueue()
    results = SimpleQueue()

    images_path = glob.glob('dataset/data/*')

    start_jobs(cpu_count(), jobs, results, images_path)

    return get_uuid_images(images_path, results)
