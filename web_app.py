import os
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

# from keras.api.preprocessing.image import ImageDataGenerator
from keras.src.legacy.preprocessing.image import ImageDataGenerator
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
import io
import streamlit as st
from PIL import Image
from keras.api.applications.efficientnet import preprocess_input, decode_predictions

# путь к файлу обученной модели
model_file_name = r"C:\MyPythonProjects\saved_models\model_dogs_4.keras"
# путь к json файлу, в котором сохранены параметры обученной модели
class_model_file_name = model_file_name + ".class"
model = load_model(model_file_name)
st.set_page_config(page_title="Определение брака", page_icon="🔩", layout="centered")
st.title("Определение брака по фото")


def load_image():
    uploaded_file = st.file_uploader(label="Выберите фото")
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


img_path = load_image()


# Функция для предсказания местоположения
def predict_location():
    # img = image.load_img(img_path, target_size=(model_data["img_width"], model_data["img_height"]))
    img = img_path
    img = img.resize((model_data["img_width"], model_data["img_height"]))
    img_array = image.img_to_array(img)
    img_array = (
        np.expand_dims(img_array, axis=0) / model_data["rescaleN"]
        - model_data["rescaleA"]
    )
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[
        0
    ]  # Получаем индекс предсказанного класса
    return predicted_class_index  # Возвращаем индекс класса


# def preprocess_image(img):
#     img = img.resize((150, 150))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)  # Убедитесь, что здесь используется корректный метод
#     return x


# def predict_location(img_array):
#     img_array /= model_data["rescaleN"]
#     img_array -= model_data["rescaleA"]
#     prediction = model.predict(img_array)
#     predicted_class_index = np.argmax(prediction, axis=1)[0]
#     return predicted_class_index


# def predict_location(img_path):
#     img = image.load_img(img_path, target_size=(model_data["img_width"], model_data["img_height"]))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / model_data["rescaleN"]-model_data["rescaleA"]
#     prediction = model.predict(img_array)
#     predicted_class_index = np.argmax(prediction, axis=1)[0]  # Получаем индекс предсказанного класса
#     return predicted_class_index  # Возвращаем индекс класса


with open(class_model_file_name, "r") as file:
    model_data = json.load(file)

result = st.button("Проанализировать фотографию")
if result:
    if img_path is not None:
        # x = preprocess_image(img)  # Применяем предобработку
        predicted_class_index = predict_location()  # Предсказание
        predicted_label = model_data["class_list"][predicted_class_index]
        st.write("Идет загрузка...")
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.05)  # Имитация загрузки
            progress_bar.progress(percent_complete + 1)
        st.success(f"На фотографии: {predicted_label}")
    else:
        st.warning("Пожалуйста, загрузите изображение.")
