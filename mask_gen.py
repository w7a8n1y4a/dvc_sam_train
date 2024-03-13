import os
import cv2
import numpy
import glob
import uuid
import shutil

from multiprocessing import queues, Process, SimpleQueue, cpu_count


def split_img_box(uuid, img, output_dir, width, height, overlap):

    rows = int((img.shape[0] / (height - height * overlap)))
    cols = int((img.shape[1] / (width - width * overlap)))

    for i in range(rows):
        for j in range(cols):
            x = int(j * (width - width * overlap))
            y = int(i * (height - height * overlap))

            crop_img = img[y:y+height, x:x+width]

            if crop_img.shape[0] == height and crop_img.shape[1] == width:
                output_image_path = os.path.join(output_dir, f'{str(uuid)}_{i}_{j}.png')
                cv2.imwrite(output_image_path, crop_img)


def split_img_devider(uuid, img, output_dir, devider, overlap):
    
    devider_overlap = int(devider/overlap)

    height = int(img.shape[0]/devider)
    width = int(img.shape[1]/devider)

    cols = int(img.shape[0] / (img.shape[0]/devider_overlap))
    rows = int(img.shape[1] / (img.shape[1]/devider_overlap))

    for i in range(cols):
        for j in range(rows):
            x = int(j * (width - width * overlap))
            y = int(i * (height - height * overlap))
           
            crop_img = img[y:y+height, x:x+width]
            
            if crop_img.shape[0] == height and crop_img.shape[1] == width:
                output_image_path = os.path.join(output_dir, f'{str(uuid)}_{i}_{j}.png')
                cv2.imwrite(output_image_path, crop_img)


def mask_gen(filepath: str) -> str:
    
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)                                                                          
    assert img is not None, "file could not be read, check with os.path.exists()"                                           
                                                                                                                              
    img = cv2.medianBlur(img, 5)                                                                                               
    ret, th1 = cv2.threshold(img, numpy.median(img), 255, cv2.THRESH_BINARY)                                                     
    
    img_uuid = uuid.uuid4()

    cv2.imwrite(f'img/{img_uuid}.png', img)
    cv2.imwrite(f'mask/{img_uuid}.png', th1)
    
    print('start', img_uuid)

    split_img_box(img_uuid, img, './dataset/train/images', 1800, 1800, 0.1)
    split_img_box(img_uuid, th1, './dataset/train/masks', 1800, 1800, 0.1)

    print("end", img_uuid)

    return str(img_uuid)


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

    return uuid_images


def gen_dataset(path_img, path_mask):
   
    os.makedirs(path_img, exist_ok=True)
    os.makedirs(path_mask, exist_ok=True)
        
    shutil.rmtree(path_img)
    shutil.rmtree(path_mask)

    os.makedirs(path_img, exist_ok=True)
    os.makedirs(path_mask, exist_ok=True)

    jobs = SimpleQueue()
    results = SimpleQueue()
    images_path = glob.glob('./dataset/data/*')
    start_jobs(cpu_count(), jobs, results, images_path)
    uuid_images = get_uuid_images(images_path, results)

    return uuid_images
