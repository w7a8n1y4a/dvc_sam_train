import os.path
import time

import httpx
import mlflow
import yaml

import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import DataLoader
from torch.optim import Adam
from segment_anything import build_sam_vit_b
from lora import LoRA_sam

from core import utils as utils
from dataloader import DatasetSegmentation, collate_fn
from processor import Samprocessor
from core.utils import dict_to_log, filepath_to_md5, create_dirs, dict_to_json_file_save, current_branch_name
from core.config import settings


def train(params: dict):
    if not os.path.isfile(f"./{params['start_model_name']}"):
        try:
            r = httpx.get(f"https://dl.fbaipublicfiles.com/segment_anything/{params['start_model_name']}", timeout=20)
        except:
            dict_to_log(
                {'model_load_failed': f"https://dl.fbaipublicfiles.com/segment_anything/{params['start_model_name']}"}
            )

        with open(f"./{params['start_model_name']}", 'wb') as f:
            f.write(r.content)

        dict_to_log({'model_loaded': params['start_model_name']})

    sam = build_sam_vit_b(checkpoint=f"./{params['start_model_name']}")
    sam_lora = LoRA_sam(sam, params['rank'])
    model = sam_lora.sam

    # Process the dataset
    processor = Samprocessor(model)
    train_ds = DatasetSegmentation(processor, mode="train")
    # Create a dataloader
    train_dataloader = DataLoader(train_ds, batch_size=params['batch'], shuffle=True, collate_fn=collate_fn)
    # Initialize optimize and Loss
    optimizer = Adam(model.image_encoder.parameters(), lr=params['learning_rate'], weight_decay=0)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    num_epochs = params['epochs']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Set model to train and into the device
    model.train()
    model.to(device)

    loss_scores_list = []

    for epoch in range(num_epochs):
        epoch_losses = []
        ious = []

        for i, batch in enumerate(tqdm(train_dataloader)):
            outputs = model(batched_input=batch, multimask_output=False)

            stk_gt, stk_out, iou = utils.stacking_batch(batch, outputs)
            stk_out = stk_out.squeeze(1)
            stk_gt = stk_gt.unsqueeze(1)

            ious.append(float(iou.data.cpu().numpy()[0][0][0]))

            loss = seg_loss(stk_out, stk_gt.float().to(device))

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            epoch_losses.append(loss.item())

        epoch_dict = {'epoch': epoch, 'avg_loss': round(mean(epoch_losses), 3), 'avg_iou': round(mean(ious), 3)}

        loss_scores_list.append(epoch_dict)

        dict_to_log(epoch_dict)

    dict_to_json_file_save({'train': loss_scores_list}, f'{settings.app_path}/plot/train/loss_scores_dict.json')

    run_name = yaml.safe_load(open("params.yaml"))['variable']['run_name']

    model_name = f"{run_name}_{params['rank']}.safetensors"
    model_path = f'{settings.app_path}/model/'

    create_dirs([model_path])

    sam_lora.save_lora_parameters(model_path + model_name)

    return model_name, model_path


def main():
    params = yaml.safe_load(open("params.yaml"))["train"]

    mlflow.set_tracking_uri('https://mlflow-test.pepemoss.com')
    exp_name = current_branch_name()
    run_name = yaml.safe_load(open("params.yaml"))['variable']['run_name']

    # создание эксперимента по имени ветки
    try:
        mlflow.create_experiment(exp_name)
        mlflow.set_experiment(experiment_name=exp_name)
    except:
        mlflow.set_experiment(experiment_name=exp_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)

        start_time = time.perf_counter()
        model_name, model_path = train(params)
        end_time = time.perf_counter()

        filepath = f'{settings.app_path}/model/train_data.json'

        train_dict = {
            'trained_model_name': model_name,
            'trained_model_md5': filepath_to_md5(model_path + model_name),
            'run_id': run.info.run_id,
            'train_time': round(end_time - start_time, 2),
        }

        dict_to_log(train_dict)

        dict_to_json_file_save(
            train_dict,
            filepath,
        )

    return True


if __name__ == "__main__":
    main()
