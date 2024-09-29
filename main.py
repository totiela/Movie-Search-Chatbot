import streamlit as st
from llm_utils import initialize_llm
from retrievers import initialize_retrievers
from prompts_and_chain import initialize_prompts_and_chain
from sql_agent import initialize_sql_agent
from tools import create_tools
from movie_agent import initialize_movie_agent, get_movie_agent_response
from download_db import download_and_prepare
import langchain
import os

langchain_tracing_v2 = st.secrets["LANGCHAIN_TRACING_V2"]
langchain_endpoint = st.secrets["LANGCHAIN_ENDPOINT"]
langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]
langchain_project = st.secrets["LANGCHAIN_PROJECT"]

import requests

# Тест запроса к серверу LangSmith
response = requests.get(os.getenv("LANGCHAIN_ENDPOINT"))
print(response.status_code)

# Отключаем режим отладки LangChain для повышения производительности
langchain.debug = False

# Кэшируем ресурсы, чтобы избежать повторной инициализации при каждом запросе
@st.cache_resource
def setup():
    # Загружаем базы данных и подготавливаем необходимые файлы
    download_and_prepare()
    # Инициализируем языковую модель и эмбеддер
    llm, embedder = initialize_llm()
    # Инициализируем ансамбль ретриверов для поиска фильмов
    ensemble_retriever = initialize_retrievers(embedder, "faiss_mistral-7b-v2_embed_index", "retrievers/bm25_retriever.pkl")
    # Инициализируем цепочку промптов для взаимодействия с моделью
    chain = initialize_prompts_and_chain(llm, ensemble_retriever)
    # Инициализируем SQL-агент для работы с базой данных фильмов
    agent_executor_sql = initialize_sql_agent("databases/movies_with_descriptions.db", llm)
    # Создаем набор инструментов, объединяющий цепочку промптов и SQL-агента
    tools_mix = create_tools(chain, agent_executor_sql)
    return llm, tools_mix

# Инициализируем LLM и набор инструментов через функцию setup()
llm, tools_mix = setup()

# Функция для инициализации или перезапуска агента с новой памятью
def initialize_agent():
    # Инициализируем Movie Search Agent с доступом к инструментам
    movie_agent_executor = initialize_movie_agent(llm, tools_mix)
    return movie_agent_executor

# Используем Streamlit session_state для сохранения агента и истории чата между запросами
if 'movie_agent_executor' not in st.session_state:
    st.session_state['movie_agent_executor'] = initialize_agent()

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Функция для очистки истории чата и переинициализации агента
def clear_chat():
    st.session_state['chat_history'] = []
    st.session_state['movie_agent_executor'] = initialize_agent()

# Отображаем заголовок приложения с HTML-стилизацией
st.markdown("<h1 style='text-align: center; color: #FF6347;'>🎬 Movie Search ChatBot</h1>", unsafe_allow_html=True)

# Боковая панель с примерами запросов для удобства пользователей
with st.sidebar:
    st.markdown("""
        Связаться со мной можно в [Telegram](https://t.me/Sanchez_Z_Z_z_Z)
    """)
    st.write("### Примеры запросов:")
    st.markdown("""
        - **Поиск фильма по описанию:** "Фильм, где человек после авиакрушения несколько лет выживает на острове"
        - **Поиск описания фильма по названию:** "Расскажи про фильм Титаник"
        - **Рекомендация фильма по жанру или актеру:** "Посоветуй мне фильм в жанре драма"; "Какие фильмы с Леонардо ДиКаприо ты можешь порекомендовать?"
        - **Группировка и поиск используя сложные запросы:** "Топ-5 самых популярных фильмов в жанре триллер"; "Покажи фильмы, в которых снимался Джонни Депп, и отсортируй их по рейтингу"
    """)

# Основное поле для ввода запроса
input_text = st.text_input("Введите запрос", key="input_text", placeholder="Например: Порекомендуй драму с ДиКаприо")

# Создаем колонки для кнопок "Отправить запрос" и "Очистить диалог"
col1, col2 = st.columns([1, 1], gap="small")

# Кнопка для отправки запроса
with col1:
    send_button = st.button("Отправить запрос", key="send_button", use_container_width=True)

# Кнопка для очистки чата
with col2:
    clear_button = st.button("Очистить диалог", key="clear_button", use_container_width=True)

# Обработка действий при нажатии на кнопки
if send_button and input_text:
    try:
        # Отображаем индикатор загрузки во время выполнения запроса
        with st.spinner("🍿 Агент обрабатывает запрос..."):
            # Получаем ответ от Movie Search ChatBot
            response = get_movie_agent_response(st.session_state['movie_agent_executor'], input_text)
            # Сохраняем запрос и ответ в историю чата
            st.session_state['chat_history'].append((input_text, response))
    except Exception as e:
        # В случае ошибки добавляем соответствующее сообщение в историю чата
        st.session_state['chat_history'].append((input_text, "Извините, я не смог обработать ваш запрос"))

# Обработка нажатия кнопки для очистки диалога
if clear_button:
    clear_chat()

# Ограничение на количество отображаемых сообщений
MAX_MESSAGES = 3  # Отображаем только последние 3 сообщения

# CSS для прокрутки контейнера (если нужно)
st.markdown("""
<style>
.scrollable-container {
    max-height: 400px;
    overflow-y: auto;
    padding-right: 15px;
}
</style>
""", unsafe_allow_html=True)

# Отображение заголовка диалога и создание двух колонок
col1, col2 = st.columns([3, 1])

with col1:
    st.write("### Диалог:")

# Если количество сообщений больше MAX_MESSAGES, отображаем чекбокс для показа всей истории
if len(st.session_state['chat_history']) > MAX_MESSAGES:
    with col2:
        show_full_history = st.checkbox("Показать всю историю чата")
else:
    show_full_history = False

# Отображаем диалог с пользователем в формате мессенджера
chat_container = st.container()

with chat_container:
    st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)
    
    # Если выбрано отображение полной истории, показываем все сообщения
    if show_full_history:
        for query, answer in st.session_state['chat_history']:
            # Сообщение пользователя
            st.markdown(f"<div style='text-align: right; padding: 10px; background-color: #F0F8FF; border-radius: 10px; margin-bottom: 10px;'>"
                        f"<strong>Вы:</strong> {query}</div>", unsafe_allow_html=True)
            # Ответ бота
            st.markdown(f"<div style='text-align: left; padding: 10px; background-color: #E0FFFF; border-radius: 10px; margin-bottom: 10px;'>"
                        f"<strong>Movie Search Bot:</strong> {answer}</div>", unsafe_allow_html=True)
    else:
        # Если нет, отображаем только последние N сообщений
        for query, answer in st.session_state['chat_history'][-MAX_MESSAGES:]:
            # Сообщение пользователя
            st.markdown(f"<div style='text-align: right; padding: 10px; background-color: #F0F8FF; border-radius: 10px; margin-bottom: 10px;'>"
                        f"<strong>Вы:</strong> {query}</div>", unsafe_allow_html=True)
            # Ответ бота
            st.markdown(f"<div style='text-align: left; padding: 10px; background-color: #E0FFFF; border-radius: 10px; margin-bottom: 10px;'>"
                        f"<strong>Movie Search Bot:</strong> {answer}</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
