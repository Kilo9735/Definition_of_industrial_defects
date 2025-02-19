import os
import numpy as np
# import matplotlib.pyplot as plt
# from keras.api.preprocessing.image import ImageDataGenerator
# from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.callbacks import EarlyStopping
from keras.applications import MobileNet
from keras.optimizers import Adamax
from keras.layers import BatchNormalization
import json


#пути

# путь к изображению, локацию которого хотим определить
path_foto = r"test_foto/20.jpg"
# путь к файлу обученной модели
model_file_name=r"saved_models/model_country3.keras"
# путь к json файлу, в котором сохранены параметры обученной модели
class_model_file_name=model_file_name+".class"



# получаем словарь model_data с данными модели из json по пути class_model_file_name
# данный json был сформирован на этапе обучения модели
# сохраненные параметры:
# img_width, img_height - размер картинок, который использует модель
# rescaleN, rescaleA - параметры нормализации
# class_list -  соответствие индексов в моделе наименованиям классов из json, 
# так как сохраненная модель после обучения содержит только индексы групп
# мы при сохранении модели дополнительно сохранили индексы в json
with open(class_model_file_name, 'r') as file:
    model_data = json.load(file)  # Загружаем список из JSON


# загружаем сохраненную обученную модель
model = load_model(model_file_name)


# Функция для предсказания местоположения
def predict_location(img_path):
    img = image.load_img(img_path, target_size=(model_data["img_width"], model_data["img_height"]))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / model_data["rescaleN"]-model_data["rescaleA"]
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]  # Получаем индекс предсказанного класса
    return predicted_class_index  # Возвращаем индекс класса

# Предсказываем местоположение
predicted_class_index = predict_location(path_foto)
# Получаем имя класса из индекса
predicted_label = model_data["class_list"][predicted_class_index]
print(f'Predicted location class: {predicted_label}')

