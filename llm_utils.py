import streamlit as st
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

def initialize_llm(model_name='google/gemma-2-27b-it'):
    api_key = st.secrets["NVIDIA_API_KEY"]

    if not api_key:
        raise ValueError("API ключ не найден! Проверьте файл .env")

    llm = ChatNVIDIA(model=model_name, nvidia_api_key=api_key)
    embedder = NVIDIAEmbeddings(model='nvidia/nv-embedqa-mistral-7b-v2', api_key=api_key)

    return llm, embedder
