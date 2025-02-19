import os
from PIL import Image
import warnings

#пути
path_ds = r"G:\\images_2"

# Функция для обработки предупреждений
def warning_handler(message, category, filename, lineno, file=None, line=None):
    print(f"Предупреждение: {message} в {filename}:{lineno}:{file_path}")
    #os.remove(file_path)

# Устанавливаем обработчик
warnings.showwarning = warning_handler

# Проверка всех изображений в директории
for subdir, dirs, files in os.walk(path_ds):
    for file in files:
        file_path = os.path.join(subdir, file)
        try:
            img = Image.open(file_path)
            img.verify()  # Проверка, что файл является корректным изображением
        except Exception as e:
            print(file_path)
            #os.remove(file_path)