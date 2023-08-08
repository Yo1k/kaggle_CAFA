import json
from pathlib import Path


DIR_TO_SAVE_LOGS = Path('путь до папки сохранения логов')
log_file = DIR_TO_SAVE_LOGS / 'имя файла с логами.json'


limit_of_memory = 1073741824  # 1 Гб


def log_to_json(log_file: Path, params: dict) -> None:
    '''
    Дописываем результаты обучения и параметры модели (params) в уже
    существующий файл JSON, если он не существует, то создаем новый.
    '''
    if log_file.exists():
        print(log_file.stat().st_size)
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
