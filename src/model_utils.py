'''
Модуль для построения и оценки производительности моделей.
'''
from __future__ import annotations

import gc
from collections import defaultdict
from typing import Any, Callable, Protocol, TypedDict

import numpy as np
import sklearn.metrics as skmetrics
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import RepeatedKFold
from sklearn.utils.class_weight import compute_class_weight


class CallbackParams(TypedDict):
    '''
    Класс для аннтоации типа.
    '''
    callback: type[tf.keras.callbacks.Callback]
    params: dict[str, Any]


class Metric(Protocol):
    '''
    Класс для аннтоации типа.
    '''
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray | float:
        ...


class ModelCreateCompile(Protocol):
    '''
    Класс для аннотации типа.
    '''
    def __call__(
        self,
        n_inputs: int,
        n_outputs: int
    ) -> tf.keras.Model:
        ...


class ModelEvalParams(TypedDict):
    '''
    Класс для аннтоации типа.
    '''
    data_info: dict[str, Any]
    model_params: dict[str, Any]
    scores: dict[str, Any]


def calculating_class_weights(y_true: np.ndarray) -> np.ndarray:
    '''
    Рассчет весов для случая 2-х классов: 0, 1
    в multi-label классификации.

    Возвращает результат следующей структуры:
    2D-массив c `.shape[0]` равной количеству целей multi-label классификации
    и `.shape[1]` равной 2:
    массив из двух элементов, где с индексом 0 находится вес для класса 0,
    с индексом 1 - вес для класса 1.
    '''
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight(
            'balanced',
            classes=[0, 1],
            y=y_true[:, i],
        )

    return weights


def create_callbacks(
    params: list[CallbackParams]
) -> list[tf.keras.callbacks.Callback]:
    '''
    Возвращает список `Callback` объектов,
    созданных из словаря `CallbackParams`.
    '''
    callbacks = list()
    for item in params:
        callbacks.append(
            item['callback'](**item['params'])
        )

    return callbacks


def create_callbacks_info(
    params: list[CallbackParams]
) -> list[dict[str, Any]]:
    '''
    Возвращает список словарей, содержащих названия callback классов
    и соответсвующие аргументы для их конструкторов.
    '''
    callbacks_info = list()
    for item in params:
        callbacks_info.append({
            'callback_name': item['callback'].__name__,
            'params': {**item['params']}
        })

    return callbacks_info


def get_scores_stats(
    results: dict[str, list],
    ndigits: int = 2,
) -> dict[str, float]:
    '''
    Возвращает словарь со средним значением и стандартным отклонением очков
    для всех метрик.
    '''
    stats = dict()
    for metric, score in results.items():
        stats[f'mean_{metric}'] = np.mean(score).round(ndigits)
        stats[f'std_{metric}'] = np.std(score).round(ndigits)

    return stats


def evaluate_model(
    features: np.ndarray,
    lbls: np.ndarray,
    proxy_model: ProxyFitModel,
    metrics: list[Metric],
    n_repeats: int = 1,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[
    defaultdict[str, list[float | np.ndarray]],
    list[tf.keras.callbacks.History]
]:
    '''
    Возвращает словарь с результатами вычисления метрик и список объектов
    History для каждого фолда при нескольких циклах K-fold кросс-валидации.

    Parameters
    ----------
    features: np.ndarray
        Признаки, составляющие датасет для обучения модели.

    lbls: np.ndarray
        Метки, составляющие датасет для обучения модели.

    proxy_model: ProxyFitModel
        Прокси для обучения модели и вычисления метрик на ней.

    metrics: list[Metric]
        Список метрик, используемы для оценки производительности моделей.

    n_repeats: int
        Число циклов K-fold кросс-валидации. По умолчанию 1 цикл.

    n_splits: int
        Число фолдов в K-fold кросс-валидации.

    random_state: int
        Число контролирующее рандом при использовании K-fold кросс-валидации.
        Используется для воспроизведения результатов.

    Returns:
    --------
    defaultdict[str, list[float | np.ndarray]]
        Возвращается словарь списков. Ключом является название функции,
        используемой в качестве метрики. В соответсвующем списке находятся
        результаты вычисления метрик на всех циклах на каждом фолде. Т.е. если
        было 3 цикла и 5 разбиений при кросс-фалидации, то в списке будет
        3*5=15 результатов метрик.
    list[tf.keras.callbacks.History]
        Возвращаем массив из объектов History каждого цикла
        кросс-валидации.
    '''
    results = defaultdict(list[float | np.ndarray])
    cv = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    histories = []
    for train_idx, test_idx in cv.split(features):
        x_train, x_test = features[train_idx], features[test_idx]
        y_train, y_test = lbls[train_idx], lbls[test_idx]

        proxy_model.make_model(features.shape[1], lbls.shape[1])
        if proxy_model.validation_split is None:
            history = proxy_model.fit(x_train, y_train, (x_test, y_test))
        else:
            history = proxy_model.fit(x_train, y_train)
        histories.append(history)
        y_pred = proxy_model.predict(x_test)

        for metric in metrics:
            results[metric.__name__].append(  # type: ignore
                metric(y_test, y_pred)
            )
        # Очистка объектов.
        proxy_model.clear()

    return results, histories


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float | np.ndarray:
    '''
    Возвращает результат f1_score для каждой метки.
    Предполагается случай multilabel classification с двумя классами: 0, 1.

    В случае, когда `y_pred` представлены вероятностями
    вместо значениями класса, им присваивается класс
    как в случае бинарной классификации с порогом 0.5:
    - `y_pred` >= 0.5 -> 1
    - `y_pred` < 0.5 -> 0
    '''
    # Классификация на 2 класса с прогом 0.5.
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    y_pred_set = set(np.unique(y_pred))
    if y_pred_set > {0, 1}:
        ValueError(f'{y_pred_set} возможные значения классов: 0, 1')
    return skmetrics.f1_score(y_true, y_pred, average=None)  # type: ignore


def f1_score_micro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    '''
    Возвращает результат f1_score полученный глобальным подсчетом
    общего количества истинных срабатываний,
    ложноотрицательных и ложноположительных результатов.
    Предполагается случай multilabel classification с двумя классами: 0, 1.

    В случае, когда `y_pred` представлены вероятностями
    вместо значениями класса, им присваивается класс
    как в случае бинарной классификации с порогом 0.5:
    - `y_pred` >= 0.5 -> 1
    - `y_pred` < 0.5 -> 0
    '''
    # Классификация на 2 класса с прогом 0.5.
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    y_pred_set = set(np.unique(y_pred))
    if y_pred_set > {0, 1}:
        ValueError(f'{y_pred_set} возможные значения классов: 0, 1')
    return skmetrics.f1_score(y_true, y_pred, average='micro')  # type: ignore


def get_basseline_model(n_inputs: int, n_outputs: int) -> tf.keras.Model:
    '''
    Возвращает скомпилированную модель, взятую в качестве исходного уровня.
    '''
    model = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(input_shape=(n_inputs,)),
        tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dense(units=n_outputs, activation='sigmoid'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss='binary_crossentropy',
        metrics=['binary_accuracy'],
    )
    return model


def metric_closure(
        metric_func: Callable[
            [np.ndarray, np.ndarray, tuple[Any], dict[str, Any]],
            np.ndarray | float],
        metric_name: str | None = None,
        *args: Any,
        **kwargs: Any,
) -> Metric:
    '''
    Вспомогательная функция для формирования метрик.

    Parameters:
    ----------
    metric_func: Callable[
        [np.ndarray, np.ndarray, tuple[Any], dict[str, Any]],
        np.ndarray | float
    ]
        Функция или класс с методом `__call__` используемая
        для рассчета метрики.

    metric_name: str | None
        Параметр для задания названия метрики, по умолчанию используется
        название функции `metric_func`.

    args: Any, kwargs: Any
        Параметры, которые передаются в качестве аргументов в `metric_func`.
    '''
    def closure(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray | float:
        return metric_func(y_true, y_pred, *args, **kwargs)
    if metric_name:
        closure.__name__ = metric_name
    else:
        closure.__name__ = metric_func.__name__
    return closure


class ModelCompileFabric:
    '''
    Класс для формирования фабрики, котороая производит одинаковые
    скомпилированные модели. Гиперпараметры моделей определяются
    параметрами конструктора. Подробнее о назначении параметров
    см. в документации по моделям `tf.keras`.

    Parameters:
    ----------
    activation: str
        Функция активации, используются имена функций из `tf.keras`.

    kernel_regularizer: str | None
        Функция регуляризации, используются имена функций из `tf.keras`.

    dropout: float
        Значения доли при прореживании, по умолчанию 0 - нет прореживания.
        Прореживание применяется ко всем внутренним слоям DNN
        с одинаковым значением доли.

    layers_strct: list[int] | None
        Список, задающий внутренную структуру DNN. Каждый элемент определяет
        количество нейронов на соответсвующем уровне.
        DNN имеет последовательную структуру.
        Порядок элементов списка соответсвует порядку уровней DNN.
        По умолчанию скрытая сетка 512 x 512 x 512.
    '''

    def __init__(
        self,
        activation: str = 'relu',
        kernel_regularizer: str | None = None,
        dropout: float = 0,
        layers_strct: list[int] | None = None,
        learning_rate: float = 0.001,
        loss: tf.keras.losses.Loss | None = None
    ):
        self.activation = activation
        self.dropout = dropout
        self.kernel_regularizer = kernel_regularizer
        self.layers_strct = [512]*3 if layers_strct is None else layers_strct
        self.learning_rate = learning_rate
        self.loss = (
            tf.keras.losses.BinaryCrossentropy() if loss is None
            else loss
        )

    def __create_compile(
        self,
        n_inputs: int,
        n_outputs: int,
    ) -> tf.keras.Model:
        # --- Формирование архитектуры сети.
        # Входной уровень сети.
        inputs = tf.keras.Input(shape=n_inputs)
        batch_normalization_lr = tf.keras.layers.BatchNormalization()(inputs)

        # Формирование внутренних слоев сети.
        x = batch_normalization_lr
        for units in self.layers_strct:
            x = tf.keras.layers.Dense(
                units=units,
                activation=self.activation,
                kernel_regularizer=self.kernel_regularizer,  # Регуляризация
            )(x)
            if self.dropout > 0:
                x = tf.keras.layers.Dropout(self.dropout)(x)  # Прореживание.

        # Выходной уровень сети.
        outputs = tf.keras.layers.Dense(n_outputs, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # --- Компиляция модели.
        model.compile(
            loss=self.loss,
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
            ],
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
            )
        )
        return model

    def __call__(self, n_inputs: int, n_outputs: int) -> tf.keras.Model:
        '''
        Возвращает скомпилированную модель.
        '''
        return self.__create_compile(n_inputs, n_outputs)


class ProxyFitModel:
    '''
    Класс для проксирования модели при обучении. Позволяет создавать модели с
    одинаковой структурой и их удалять с очисткой памяти. Используется,
    например, в кросс-валидации для оптимизации использования памяти.

    Parameters:
    ----------
    model_create_compile_func: ModelCreateCompile
        Функция или класс с методом `__call__` позволяющая создавать
        одинаковые скомпилированные DNN.

    validation_split: float | None
        Значения параметров отличаются от значений
        исходного метода `fit` модели `tf.keras.Model`:
        Если `None`, то метод `fit` требует аргумента `validation_data`,
        если `float`, то будет использоваться разбиение тренировочных данных
        с пропорцией `validation_split`.

    Остальные параметры в конструкторе повторяют часть параметров метода `fit`
    модели `tf.keras.Model`.
    '''
    def __init__(
        self,
        model_create_compile_func: ModelCreateCompile,
        *,
        batch_size: int = 1000,
        callbacks: list[tf.keras.callbacks.Callback] | None = None,
        class_weight: dict[int, float] | None = None,
        epochs: int = 10,
        shuffle: bool = True,
        steps_per_epoch: float | None = None,
        validation_split: float | None = 0.0,
        verbose: str | int = 'auto',
    ):
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.class_weight = class_weight
        self.epochs = epochs
        self.model_create_compile_func = model_create_compile_func
        self.shuffle = shuffle
        self.steps_per_epoch = steps_per_epoch
        self.validation_split = validation_split
        self.verbose = verbose
        self.__model: tf.keras.Model | None = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tf.keras.callbacks.History:
        '''
        Обучение модели.
        '''
        assert self.__model is not None, (
            'Необходимо создать модель, вызовите метод `make_model`'
        )
        if self.validation_split is not None and validation_data is not None:
            raise ValueError(
                'Недопустимо одновременно применение разбиения данных '
                'через `validation_split` на тренировочные и валидационные '
                'данные при наличии `validation_data`'
                'Необходимо установить `validation_split=None` '
                'или `validation_data=None`'
            )

        return self.__model.fit(
            x=x,
            y=y,
            batch_size=self.batch_size,
            callbacks=self.callbacks,
            class_weight=self.class_weight,
            epochs=self.epochs,
            shuffle=self.shuffle,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=validation_data,
            validation_split=self.validation_split,  # type: ignore
            verbose=self.verbose,  # type: ignore
        )

    def clear(self) -> None:
        '''
        Удаление модели и очистка памяти.
        '''
        self.__model = None
        gc.collect()
        tf.keras.backend.clear_session()

    def make_model(self, n_inputs: int, n_outputs: int) -> None:
        '''
        Создать и скомпилировать модель.
        '''
        self.__model = self.model_create_compile_func(n_inputs, n_outputs)

    def predict(self, x: np.ndarray) -> np.ndarray:
        '''
        Возвращает результат предикции обученной модели.
        '''
        assert self.__model is not None, (
            'Необходимо создать модель, вызовите метод `make_model`'
        )
        return self.__model.predict(x)

    @property
    def original_model(self) -> tf.keras.Model | None:
        return self.__model


class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    '''
    Рассчитывает взвешенную `tf.keras.losses.BinaryCrossentropy`.
    Подробнее о параметрах конструктора см.
    в документации `tf.keras.losses.BinaryCrossentropy`.

    `class_weights`: None | np.ndarray
    - В случае `None` возвращается результаты вычисления без взвешивания.
    - В случае `np.ndarray`, структура должна быть следующая:
    2D-массив c `.shape[0]` равной количеству целей multi-label классификации
    и `.shape[1]` равной 2:
    массив из двух элементов, где с индексом 0 находится вес для класса 0,
    с индексом 1 - вес для класса 1.
    '''
    def __init__(
        self,
        class_weights: None | np.ndarray = None,
        from_logits: bool = False,
        label_smoothing: float = 0.0,
        name: str | None = 'weighted_binary_crossentropy',
        axis=-1,
    ):
        # Случай когда loss функция сводится к binary_crossentropy.
        if class_weights is None or np.all(class_weights == 1.0):
            self.class_weights = None
            name = 'binary_crossentropy'
        else:
            self.class_weights = tf.convert_to_tensor(
                class_weights,
                dtype=tf.float32
            )
        self.default_bce = tf.keras.losses.BinaryCrossentropy(
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
        )
        super(WeightedBinaryCrossentropy, self).__init__(name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = self.default_bce(y_true, y_pred, sample_weight)
        if self.class_weights is not None:
            y_true = tf.cast(y_true, tf.float32)
            loss = K.mean(
                (
                    (self.class_weights[:, 0]**(1-y_true))  # type: ignore
                    * (self.class_weights[:, 1]**(y_true))  # type: ignore
                    * self.default_bce(y_true, y_pred)
                ),
                axis=-1
            )

        return loss
