import os
import sqlite3
from translate import Translator

# Укажите путь к папке, в которой нужно переименовать папки
folder_path = "G:\Dogs"
# Определение с какого на какой язые мы переводим
translator = Translator(from_lang="russian", to_lang="en")


# Функция для удаления первых 10 символов из названия папки
def rename_folders(folder_path):
    # Подключение к базе данных (или создание, если её нет)
    conn = sqlite3.connect("dogs.db")
    cursor = conn.cursor()

    # Создание таблицы, если она не существует
    # cursor.execute('''
    #     CREATE TABLE IF NOT EXISTS renamed_folders (
    #         id INTEGER PRIMARY KEY AUTOINCREMENT,
    #         old_name TEXT NOT NULL,
    #         new_name TEXT NOT NULL
    #     )
    # ''')

    # Сканирование папки
    for folder_name in os.listdir(folder_path):
        folder_full_path = os.path.join(folder_path, folder_name)

        # Проверка, является ли это папкой
        if os.path.isdir(folder_full_path):
            # Удаление первых 10 символов из названия
            new_name = translator.translate(folder_name)

            # Новый полный путь
            new_full_path = os.path.join(folder_path, new_name)

            # Переименование папки
            os.rename(folder_full_path, new_full_path)

            # Сохранение данных в базу данных
            cursor.execute(
                """
                INSERT INTO renamed_folders (old_name, new_name)
                VALUES (?, ?)
            """,
                (folder_name, new_name),
            )

            print(f"Переведено: {folder_name} -> {new_name}")

    # Сохранение изменений в базе данных и закрытие соединения
    conn.commit()
    conn.close()


rename_folders(folder_path)
