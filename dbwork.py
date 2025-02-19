import sqlite3 
import pandas as pd 
import httplib2
import os
import urllib.request
  
# https://huggingface.co/datasets/visheratin/google_landmarks_places  
# https://github.com/cvdfoundation/google-landmark?tab=readme-ov-file

#пути  
db_path=r'country.db'
directory_path = r'/Volumes/MyUSB/images'  
cache_path=r'/Volumes/MyUSB/.cache'

#sql запросы
sql_country="SELECT DISTINCT country FROM second"
#sql_main="select second.id ,country, url from second join first on(second.id=first.landmark_id) where country not in('United States','United Kingdom','Germany','France','Italy','Spain','Canada','Japan','Poland','Austria','Czechia','Russia','Netherlands','China','Austria','Sweden','Israel','India','Switzerland') order by first.id LIMIT -1 OFFSET 30100 "

#sql_main="select second.id ,country, url from second join first on(second.id=first.landmark_id) join country on (country.name=second.country and country.num_files>600 and country.num_files<900) order by first.id "        
sql_main="select second.id ,country, url from second join first on(second.id=first.landmark_id) join country on (country.name=second.country and country in ('Finland','Singapore','Egypt')) order by first.id "  
# Connect to SQLite database 
conn = sqlite3.connect(db_path) 
  
# перегоняем таблицы в бд
# stud_data = pd.read_csv('csv/first.csv') 
# stud_data.to_sql('first', conn, if_exists='replace', index=False) 

# stud_data = pd.read_csv('csv/second.csv') 
# stud_data.to_sql('second', conn, if_exists='replace', index=False) 
##################################

#получаем все страны и создаем по ним директории
#Create a cursor object 
cur = conn.cursor() 
cur.execute("PRAGMA read_uncommitted = true;")
# Fetch and display result 
cur.execute(sql_country)
result=cur.fetchall()
for row in result:
    directory = directory_path +r'/'+ row[0]  # Получаем название страны
    os.makedirs(directory, exist_ok=True)  # Создаем директорию, если она не существует

count=0
#получаем имя страны и url картинки в курсор
cur.execute(sql_main)
result=cur.fetchall()
cur.close()
conn.close() 

for row in result:
    #скачиваем картинку по url и сохраняем ее в диркторию
    count += 1 
    id = row[0]  # id
    country = row[1]      # country
    url = row[2]  
    file_name = os.path.basename(url)
    directory =directory_path+ f'/{country}'
    out_path = os.path.join(directory, f'{file_name}') 
    print(f"url: {url}")
    if os.path.exists(out_path):
        print(f"{count} пропускаем: {out_path}")
        continue
    try:
      h = httplib2.Http(cache_path)
      response, content = h.request(url)
    except Exception as e:
      print(f"ошибка: {e}")  
      continue
    # Проверяем, успешно ли выполнен запрос
    if response.status == 200:
        try:
          print(f"{count} Сохраняем изображение: {out_path}")
          with open(out_path, 'wb') as out:
              out.write(content)
        except Exception as e:
          print(f"ошибка: {e}")  
          continue
    else:
        print(f"Не удалось скачать изображение по URL: {url}, статус: {response.status}") 
   
