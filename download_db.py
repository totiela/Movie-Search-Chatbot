import gdown
import os

# Функция для скачивания файлов и подготовки необходимых папок
def download_and_prepare():
    # Определяем путь для файлов FAISS
    faiss_dir = 'faiss_mistral-7b-v2_embed_index'
    os.makedirs(faiss_dir, exist_ok=True)  # Создаем папку, если она не существует

    # Определяем путь для файлов базы данных и BM25 ретривера
    db_dir = 'databases'
    retriever_dir = 'retrievers'
    os.makedirs(db_dir, exist_ok=True)  # Создаем папку для базы данных
    os.makedirs(retriever_dir, exist_ok=True)  # Создаем папку для ретривера

    # Ссылки на файлы, которые нужно скачать, и их местоположение на локальном диске
    files_to_download = {
        # Файлы для FAISS
        "https://drive.google.com/uc?id=1w7b_I5hZcWkg_MrC5la2I29b8g3-l_lS": os.path.join(faiss_dir, "index.pkl"),
        "https://drive.google.com/uc?id=1WzHrCstqNslBeQxOakL51Z4maLip464w": os.path.join(faiss_dir, "index.faiss"),
        
        # Файл базы данных SQLite
        "https://drive.google.com/uc?id=1wEM2oMvbBkZ5-xtTznsHWScFVbCL8RCU": os.path.join(db_dir, "movies_with_descriptions.db"),
        
        # Файл для BM25 ретривера
        "https://drive.google.com/uc?id=1xbpYAHCo0KcwULTeBmylVsULMeaqWDZx": os.path.join(retriever_dir, "bm25_retriever.pkl"),
    }

    # Проходим по каждому файлу и проверяем, существует ли он локально
    for url, output in files_to_download.items():
        if not os.path.exists(output):
            print(f"Скачиваем {output}...")  # Сообщение о начале скачивания
            gdown.download(url, output, quiet=False)  # Скачивание файла через gdown
        else:
            print(f"Файл {output} уже существует, пропускаем скачивание.")  # Сообщение о пропуске, если файл уже скачан
