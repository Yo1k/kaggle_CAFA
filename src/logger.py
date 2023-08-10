import json
from pathlib import Path
import pandas as pd


def log_to_json(
    log_file: Path,
    params: dict,
    limit_of_memory=1_073_741_824
) -> None:
    '''
    Дописываем результаты обучения и параметры модели (params) в уже
    существующий файл JSON, если он не существует, то создаем новый.

    Ограничение на размер файла задано 1 Гб, тк файл для добавления полностью
    загружается в память.
    '''
    if log_file.exists():
        if log_file.stat().st_size > limit_of_memory:
            print('Ваш файл слишком большой, пора начать новый')
        with open(log_file, 'r') as f:
            data = json.load(f)
        data.update(params)
        result = 'дополнен'
    else:
        data = params
        result = 'создан'
    with open(log_file, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        print(f'Файл был {result}')


def log_to_table(log_file: Path) -> pd.DataFrame:
    '''
    Создаем из данных логирования обучения таблицу со столбцами:
        - название модели
        - тип регуляризации
        - структура слоев
        - использовался ли `callback`
        - прореживание
        - средне квадратичная ошибка метрики `f1_micro`
        - стандартное отклонение метрики `f1_micro`
    '''
    with open(log_file, 'r') as f:
        models_dict = json.load(f)

    df = pd.DataFrame(columns=[
        'model_name', 'kernel_regularizer', 'layers_strct', 'callback',
        'dropout', 'mean_f1_score_micro', 'std_f1_score_micro']
    )

    for model_name in models_dict.keys():
        kernel_regularizer = models_dict[model_name]['model_params'][
            'kernel_regularizer'
        ]
        layers_strct = models_dict[model_name]['model_params']['layers_strct']
        if models_dict[model_name]['model_params']['callbacks'] is not None:
            callback = models_dict[model_name]['model_params']['callbacks'][0][
                'callback_name'
            ]
        else:
            callback = None
        dropout = models_dict[model_name]['model_params']['dropout']
        mean_f1_score_micro = models_dict[model_name]['scores'][
            'mean_f1_score_micro'
        ]
        std_f1_score_micro = models_dict[model_name]['scores'][
            'std_f1_score_micro'
        ]
        df.loc[len(df)] = [
            model_name, kernel_regularizer, layers_strct, callback, dropout,
            mean_f1_score_micro, std_f1_score_micro
        ]

    return df
