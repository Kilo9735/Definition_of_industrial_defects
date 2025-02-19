import os
import numpy as np
import matplotlib.pyplot as plt

# from keras.api.preprocessing.image import ImageDataGenerator
from keras.api.applications import ResNet50
from keras.api.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.api.models import Model, load_model
from keras.api.optimizers import Adam
from keras.api.preprocessing import image
from keras.api.callbacks import EarlyStopping
from keras.api.applications import MobileNet
from keras.api.optimizers import Adamax
from keras.api.layers import BatchNormalization
import json
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint  # NEW

from sklearn.utils.class_weight import compute_class_weight  # NEW

# import os
# import numpy as np
# import matplotlib.pyplot as plt
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# from keras.api.applications import MobileNet
from keras.api.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.api.models import Model
from keras.api.optimizers import Adamax
import json


# НАША МОДЕЛЬ


# пути
# путь к файлу для сохранения обученной модели
model_file_name = r"C:\MyPythonProjects\saved_models\model_dogs_6.keras"
# путь к файлу для сохранения параметров обученной модели
class_model_file_name = model_file_name + ".class"
# путь к директории с изображениями для обучения модели
path_ds = r"G:\Dogs"


''# ТИПО УЛУЧШЕННАЯ МОДЕЛЬ , НО ОНА ОТКАЗЫВАЕТСЯ РАБОТАТЬ С ПЕРВЫМ И ВТОРЫМ ЭТАПОМ ОБУЧЕНИЯ
early_stopping = EarlyStopping(monitor="val_loss", patience=3, verbose=1)
# Параметры
img_width, img_height = 224, 224  # NEW: Увеличено для MobileNet
batch_size = 64
epochs = 50  # NEW: Увеличено количество эпох

# NEW: Правильные параметры для MobileNet
rescaleN = 127.5
rescaleA = 1

# NEW: Улучшенная аугментация данных
train_datagen = ImageDataGenerator(
    rescale=1 / rescaleN - rescaleA,  # Для 224x224
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

# NEW: Добавлен seed для воспроизводимости
train_generator = train_datagen.flow_from_directory(
    path_ds,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    seed=42,  # NEW
)

validation_generator = train_datagen.flow_from_directory(
    path_ds,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42,  # NEW
)

# NEW: Балансировка классов
class_weights = compute_class_weight(
    "balanced", classes=np.unique(train_generator.classes), y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# Архитектура модели
base_model = MobileNet(
    include_top=False,
    input_shape=(img_width, img_height, 3),
    pooling="max",
    weights="imagenet",
    dropout=0.4,  # я все изменил
)

x = base_model.output
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(rate=0.3, seed=123)(x)
output = Dense(train_generator.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# NEW: Callbacks
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True  # Увеличено терпение
)

lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6
)

checkpoint = ModelCheckpoint(
    "best_model.keras", monitor="val_accuracy", save_best_only=True, mode="max"
)

# Компиляция
model.compile(
    Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"]
)

# Первый этап обучения (замороженные слои)
# print("Первый этап обучения (замороженные слои)")
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // batch_size,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // batch_size,
#     epochs=epochs,
#     callbacks=[early_stopping, lr_scheduler, checkpoint],  # NEW
#     class_weight=class_weights,  # NEW
# )

# NEW: Второй этап обучения (разморозка части слоев)
# print("\nВторой этап обучения (разморозка слоев)")
# for layer in base_model.layers[-20:]:  # Размораживаем последние 20 слоев
#     layer.trainable = True

# model.compile(
#     Adamax(learning_rate=1e-5),  # Меньший learning rate
#     loss="categorical_crossentropy",
#     metrics=["accuracy", "precision", "recall"],
# )
try:
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs,
        callbacks=[early_stopping, lr_scheduler, checkpoint],
    )
except Exception as e:
    print(f"Ошибка при обучении модели: {e}")

# Сохранение модели
model.save(model_file_name)

# Сохранение метаданных
model_data = {
    "img_width": img_width,
    "img_height": img_height,
    "rescaleN": rescaleN,
    "rescaleA": rescaleA,
    "class_list": list(train_generator.class_indices.keys()),
}

with open(class_model_file_name, "w") as file:
    json.dump(model_data, file)

# Шаг 5: Оценка модели
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation loss: {loss}, Validation accuracy: {accuracy}")


# Шаг 6: Визуализация метрик
plt.figure(figsize=(15, 5))

# График точности (accuracy)
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy Dynamics")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()

# График потерь (loss)
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss Dynamics")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.show()


print(f"Обучение модели завершено!!!")

"""# Параметры
img_width, img_height = (
    160,
    160,
)  # размер, к которому приводятся изображения при загрузке для обучения модели
batch_size = 128  # число изображений для загрузки в каждой итерации - скорость обучения
epochs = 50  # Увеличить количество эпох при необходимости

# Когда загружаем изображения для обучения нейронной сети, пиксели изображений обычно имеют значения в диапазоне от 0 до 255 (в случае RGB изображений).
# Однако многие модели лучше работают, когда входные данные нормализованы, то есть приведены к диапазону от 0 до 1 или от -1 до 1.
# модель MobileNet работает с диапазоном от -1 до 1. соответственно (1/157.5)-1
# если модель работает с диапазоном от 0 до 1 то 1/255
# соответсвенно получим формулу 2 переменные для использования в ImageDataGenerator для обучения модели
# и для использования в предсказании по сохраненной модели
# эти две переменные мы так же сохраним совместно с другими данными модели для последующего использования в предсказании
rescaleN = 157.5
rescaleA = 1
# rescaleN=1255 , rescaleA=0 # для другого типа модели на всякий

# Шаг 1: Подготовка данных
train_datagen = ImageDataGenerator(
    rescale=1 / rescaleN - rescaleA, 
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

# генератор для обучения
train_generator = train_datagen.flow_from_directory(
    path_ds,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
)

# генератор для валидации
validation_generator = train_datagen.flow_from_directory(
    path_ds,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False,  # Если параметр shuffle установлен в True, данные будут случайным образом перемешаны перед каждой эпохой обучения
)

# Создание модели
base_model = MobileNet(
    include_top=False,
    input_shape=(img_width, img_height, 3),
    pooling="max",
    weights="imagenet",
    dropout=0.4,
)
x = base_model.output
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(rate=0.3, seed=123)(x)
output = Dense(train_generator.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

#  Замораживаем слои базовой модели
for layer in base_model.layers:
    layer.trainable = False

# Компиляция модели
model.compile(Adamax(), loss="categorical_crossentropy", metrics=["accuracy"])


# инструмент для остановки обучение модели, если она не показывает улучшения на валидационном наборе данных в течение определенного количества эпох (patience)
# verbose=1 - выводит сообщение при раннем завершении (verbose=0 - не выводит )
early_stopping = EarlyStopping(monitor="val_loss", patience=3, verbose=1)

# Шаг 4: Обучение модели
# train_generator.samples -общее количество образцов (изображений) в обучающем наборе наборе данных
# validation_generator.samples -общее количество образцов (изображений) в валидационном наборе данных
try:
    history = model.fit(
        train_generator,  # генератор для обучения, который будет подавать данные на вход модели
        steps_per_epoch=train_generator.samples
        // batch_size,  # количество шагов (или итераций), которые модель будет выполнять за одну эпоху обучения
        validation_data=validation_generator,  # генератор для валидации, который будет использоваться для оценки производительности модели на валидационных данных после каждой эпохи
        validation_steps=validation_generator.samples
        // batch_size,  # количество шагов, которые модель будет выполнять для валидации
        epochs=epochs,  # сколько раз модель будет проходить через весь обучающий набор данных
        callbacks=[early_stopping],  # Добавляем early stopping
    )
except Exception as e:
    print(f"Ошибка при обучении модели: {e}")

# сохраняем обученную модель
model.save(model_file_name)

# упаковываем данные в словарь, которые не сохраняются при сохранении самой обученной модели
# но используются в предсказании по загруженной модели
# далее мы этот словарь засериализуем в json
model_data = {
    "img_width": img_width,
    "img_height": img_height,
    "rescaleN": rescaleN,
    "rescaleA": rescaleA,
    "class_list": list(train_generator.class_indices.keys()),
}

# сохраняем соответствие индексов в моделе наименованиям классов в json,
# так как сохраненная модель после обучения содержит только индексы групп
with open(class_model_file_name, "w") as file:
    json.dump(model_data, file)  # Сериализуем список в JSON формат


# Шаг 5: Оценка модели
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation loss: {loss}, Validation accuracy: {accuracy}")

# Шаг 6: Визуализация метрик
plt.figure(figsize=(15, 5))

# График точности (accuracy)
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy Dynamics")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()

# График потерь (loss)
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss Dynamics")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.show()


print(f"Обучение модели завершено!!!")"""
