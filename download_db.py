import gdown
import os

# Скачивание и создание папок
def download_and_prepare():
    # Путь для файлов FAISS
    faiss_dir = 'faiss_mistral-7b-v2_embed_index'
    os.makedirs(faiss_dir, exist_ok=True)

    # Путь для файлов базы данных и BM25 ретривера
    db_dir = 'databases'
    retriever_dir = 'retrievers'
    os.makedirs(db_dir, exist_ok=True)
    os.makedirs(retriever_dir, exist_ok=True)

    # Ссылки на файлы
    files_to_download = {
        "https://drive.google.com/uc?id=1w7b_I5hZcWkg_MrC5la2I29b8g3-l_lS": os.path.join(faiss_dir, "index.pkl"),
        "https://drive.google.com/uc?id=1WzHrCstqNslBeQxOakL51Z4maLip464w": os.path.join(faiss_dir, "index.faiss"),
        "https://drive.google.com/uc?id=1wEM2oMvbBkZ5-xtTznsHWScFVbCL8RCU": os.path.join(db_dir, "movies_with_descriptions.db"),
        "https://drive.google.com/uc?id=1xbpYAHCo0KcwULTeBmylVsULMeaqWDZx": os.path.join(retriever_dir, "bm25_retriever.pkl"),
    }

    # Скачивание файлов, если их еще нет
    for url, output in files_to_download.items():
        if not os.path.exists(output):
            print(f"Скачиваем {output}...")
            gdown.download(url, output, quiet=False)
        else:
            print(f"Файл {output} уже существует, пропускаем скачивание.")