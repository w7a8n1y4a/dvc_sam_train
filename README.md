# Обучение SAM c использованием LoRa

## Конфигурации [Nginx](https://git.pepemoss.com/config/all/-/tree/master/nginx/main?ref_type=heads) для зависимостей

## Репозиторий [Backend](https://git.pepemoss.com/universitat/ml/sam_train_backend.git) - интеграция модели 

## Базовые настройки для разработки

0. Интерпретатор `>=3.10,<3.13`
1. Поддержать код в хорошем состоянии `black  ./src  -l 120 --target-version py310 -S`
2. `.env_example` - содержит базовые настройки для отправки данных в [MonkeyFilesRobot](https://t.me/MonkeyFilesRobot)
3. `.dvc/config_example.local` - требуется переместить в `.dvc/config.local`, предоставляет доступ только для чтения по умолчанию

## Для запуска без контейнера

0. Установить `poetry install`
1. Войти в окружение `poetry shell`
2. `dvc pull`
3. `dvc repro`

## Датасет

- `RGB` спутниковые снимки размером 10000x10000 пикселей в радио диапазоне

## Pipeline DVC

### Split

- Создаёт датасет при помощи разделения снимков `10000x10000` на снимки размером `1024x1024` с разным перекрытием. Разделяет датасет на `Train` и `Test`. Скрипт разрезки работает многопоточно, при помощи библиотеки `multiprocessing` с использованием простых очередей, при этом используются все доступные потоки

### Train

- Обучает модель, записывает состояния эпох в `plot/train/loss_scores_dict.json`

### Evaluate

- Считает внутренний `DiceLoss` и метрику `IoU` для тестовой выборки. `IoU` для `SAM` эквивалентен `confidence`. На малых размерах выборки `IoU` из-за особенностей расчёта может незначительно превышать 1

### Infer

- Проводит инференц для трёх случайных изображений из `Test`. И отправляет сравнительную информацию в [MonkeyFilesRobot](https://t.me/MonkeyFilesRobot)

# Частично вдохновлено

1. https://github.com/WangRongsheng/SAM-fine-tune/tree/main
