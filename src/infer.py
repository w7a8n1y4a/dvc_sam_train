import glob
import random

import torch
import numpy as np
import yaml

from segment_anything import build_sam_vit_b, SamPredictor
from lora import LoRA_sam
import matplotlib.pyplot as plt
import core.utils as utils
from PIL import Image, ImageDraw

from core.config import settings


def infer(model, params, image_path, filename, mask_path=None):

    image = Image.open(image_path)
    four = Image.open('random_image.png')

    mask = Image.open(mask_path)
    ground_truth_mask = np.array(mask)
    box = utils.get_bounding_box(ground_truth_mask)

    predictor = SamPredictor(model)
    predictor.set_image(np.array(image))
    masks, iou_pred, low_res_iou = predictor.predict(
        box=np.array(box),
        multimask_output=False, 
    )

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(15, 15))

    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline ="red")
    ax1.imshow(image)
    ax1.set_title(f"Original image + Bounding box: {filename}")

    ax2.imshow(ground_truth_mask)
    ax2.set_title(f"Ground truth mask: {filename}")

    ax3.imshow(masks[0])

    ax3.set_title(f"SAM LoRA rank {params['train']['rank']} prediction: {filename}")

    ax4.imshow(four)
    ax4.set_title(f"Noice")

    plt.savefig(f"plot/infer/{filename}.png")


def main():
    params = yaml.safe_load(open("params.yaml"))

    # получаем данные о модели и о id запуска
    model_json_path = f'{settings.app_path}/model/train_data.json'
    train_data_dict = utils.json_file_to_dict(model_json_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = build_sam_vit_b(checkpoint=f"./{params['train']['start_model_name']}")
    rank = 512
    sam_lora = LoRA_sam(sam, rank)
    sam_lora.load_lora_parameters(f"./model/{train_data_dict['trained_model_name']}")
    model = sam_lora.sam

    model.eval()
    model.to(device)

    images_path = glob.glob("split/test/images/*.png")

    random.shuffle(images_path)

    list_names = ['one', 'two', 'three']

    utils.create_dirs(['plot/infer/'])

    for name, item in zip(list_names, images_path[:3]):
        infer(
            model,
            params,
            f"split/test/images/{item.split('/')[-1]}",
            name,
            mask_path=f"split/test/masks/{item.split('/')[-1]}"
        )

        utils.dict_to_log({'name': name, 'img_name': item})

        utils.dict_to_log({'send_succes': str(utils.send_file_to_telegram(f"{settings.app_path}/plot/infer/{name}.png"))})


if __name__ == "__main__":
    main()
