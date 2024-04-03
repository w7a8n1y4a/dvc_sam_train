import shutil
import yaml

import mlflow
import torch
from tqdm import tqdm
from statistics import mean
from torch.utils.data import DataLoader
import monai

from core.config import settings
from core.utils import json_file_to_dict, dict_to_json_file_save, dict_to_log, send_file_to_telegram

from dataloader import DatasetSegmentation, collate_fn
from processor import Samprocessor
from segment_anything import build_sam_vit_b
from lora import LoRA_sam


def evaluate(model_name: str) -> dict:

    params = yaml.safe_load(open("params.yaml"))["train"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    with torch.no_grad():
        sam = build_sam_vit_b(checkpoint=f"./{params['start_model_name']}")
        sam_lora = LoRA_sam(sam, params['rank'])
        sam_lora.load_lora_parameters(f"./model/{model_name}")  # todo поменять
        model = sam_lora.sam

        processor = Samprocessor(model)
        dataset = DatasetSegmentation(processor, mode="test")
        test_dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

        model.eval()
        model.to(device)

        total_loss = []
        total_iou = []
        for i, batch in enumerate(tqdm(test_dataloader)):
            outputs = model(batched_input=batch,
                            multimask_output=False)

            gt_mask_tensor = batch[0]["ground_truth_mask"].unsqueeze(0).unsqueeze(0)
            loss = seg_loss(outputs[0]["low_res_logits"], gt_mask_tensor.float().to(device))

            iou = outputs[0]["iou_predictions"]

            total_loss.append(loss.item())
            total_iou.append(iou.cpu().numpy()[0][0])

    return {'dice_loss': float(mean(total_loss)), 'iou': float(mean(total_iou))}


def main():
    # получаем данные о модели и о id запуска
    model_json_path = f'{settings.app_path}/model/train_data.json'
    train_data_dict = json_file_to_dict(model_json_path)

    # удаляет старые запуски val от ultralytics
    try:
        shutil.rmtree(f'{settings.app_path}/runs/detect/val')
    except FileNotFoundError:
        dict_to_log({'info': "No alter val"})

    # устанавливаем путь до mlflow инстанса todo refactor убрать статику
    mlflow.set_tracking_uri('https://mlflow-test.pepemoss.com')

    with mlflow.start_run(run_id=train_data_dict['run_id']) as run:

        metrics = evaluate(train_data_dict['trained_model_name'])

        metrics = {'train': metrics}
        metrics['train']['exp_time'] = train_data_dict["train_time"]

        # сохраняем метрику в файл
        filepath = f'{settings.app_path}/eval/metrics.json'
        dict_to_log({'test_metrics': metrics})
        dict_to_json_file_save(metrics, filepath)

        # отправляем метрику в mlflow
        mlflow.log_metrics({key.replace('(', '').replace(')', ''): value for key, value in metrics['train'].items()})

        # загрузка модели в mlflow, работает только как ра
        shutil.copy(
            f'{settings.app_path}/model/{train_data_dict["trained_model_name"]}',
            f'{settings.app_path}/model/best.safetensors'
        )
        mlflow.log_artifact(f'{settings.app_path}/model/best.safetensors')

    return True


if __name__ == "__main__":
    main()
