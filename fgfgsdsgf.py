
import torch
import tensorflow as tf
import ctypes
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = 'DEBUG'  # Установите уровень логов на DEBUG
print(sys.version)


#Проверка видит ли Pytorch GPU
print("GPU доступен для Pytorch:", torch.cuda.is_available())
print("версия tensorflow: ", tf.__version__)


# 
cudnn = ctypes.WinDLL(r'cudnn64_9.dll')  
cudnn.cudnnGetVersion.restype = ctypes.c_size_t
version = cudnn.cudnnGetVersion()
print(f"cuDNN version: {version}")

# Проверка, доступен ли GPU
print(tf.config.get_visible_devices())
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("GPU доступен для Tensorflow!")
    for gpu in gpus:
        print(f"Устройство: {gpu}")
else:
    print("GPU не найден. TensorFlow будет использовать CPU.")
