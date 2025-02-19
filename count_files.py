import os
import sqlite3 


#пути
directory_path = r'G:\images'  
db_path=r'G:\country.db'
all_country_lst = {}

#sql запросы
sql_clear="DELETE FROM country"
sql_insert= "INSERT INTO country VALUES (?,?)"

def count_files_in_subdirectories(directory):
    file_counts = {}
    for root, dirs, files in os.walk(directory):
        # Считаем файлы только в подкаталогах, игнорируя корневой каталог
        if root != directory:
            file_counts[root] = len(files)
    return file_counts



file_counts = count_files_in_subdirectories(directory_path)

conn = sqlite3.connect(db_path, check_same_thread=False )
cursor = conn.cursor()

# SQL-команда для удаления всех записей из таблицы
cursor.execute(sql_clear)

# Сохраняем изменения
conn.commit()

for subdir, count in file_counts.items():
    # Убираем родительский каталог из пути
    relative_subdir = os.path.relpath(subdir, directory_path)
    all_country_lst[subdir] = count
    cursor.execute(sql_insert, (relative_subdir,count))

top_6 = sorted(all_country_lst.items(), key=lambda x: x[1], reverse=True)[:6]
# сортируем чтобы было топ 6 стран по количеству фото
for key, value in top_6:
    print(f"{key[10:]}: {value}")
    # выводим страны и количество фото убирая путь до файла

# for subdir, count in file_counts.items():
#     # Убираем родительский каталог из пути
#     relative_subdir = os.path.relpath(subdir, directory_path)
#     print(f"{relative_subdir}:   {count}")
#     cursor.execute(sql_insert, (relative_subdir,count))

# Закрываем соединение
conn.commit()
conn.close()