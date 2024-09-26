import streamlit as st
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

# Функция для инициализации LLM и эмбеддингов с использованием NVIDIA API
def initialize_llm(model_name='google/gemma-2-27b-it'):
    # Получаем API-ключ из конфигурационного файла (например, secrets.toml)
    api_key = st.secrets["NVIDIA_API_KEY"]

    # Проверяем наличие API-ключа, если его нет - выбрасываем исключение
    if not api_key:
        raise ValueError("API ключ не найден!")

    # Инициализация модели LLM от NVIDIA с использованием переданного имени модели и API-ключа
    llm = ChatNVIDIA(model=model_name, nvidia_api_key=api_key)
    
    # Инициализация эмбеддингов на основе модели NVIDIA для поиска
    embedder = NVIDIAEmbeddings(model='nvidia/nv-embedqa-mistral-7b-v2', api_key=api_key)

    # Возвращаем инициализированную LLM и модель эмбеддингов
    return llm, embedder
