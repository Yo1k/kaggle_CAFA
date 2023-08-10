import json
from pathlib import Path


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
