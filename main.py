import streamlit as st
from llm_utils import initialize_llm
from retrievers import initialize_retrievers
from prompts_and_chain import initialize_prompts_and_chain
from sql_agent import initialize_sql_agent
from tools import create_tools
from movie_agent import initialize_movie_agent, get_movie_agent_response
from download_db import download_and_prepare
import langchain

# Кэшируем сложные инициализации, чтобы они не перезапускались при каждом запросе
langchain.debug = False

@st.cache_resource
def setup():
    download_and_prepare()
    llm, embedder = initialize_llm()
    ensemble_retriever = initialize_retrievers(embedder, "faiss_mistral-7b-v2_embed_index", "retrievers/bm25_retriever.pkl")
    chain = initialize_prompts_and_chain(llm, ensemble_retriever)
    agent_executor_sql = initialize_sql_agent("databases/movies_with_descriptions.db", llm)
    tools_mix = create_tools(chain, agent_executor_sql)
    return llm, tools_mix

llm, tools_mix = setup()

# Функция для (ре)инициализации агента с новой памятью
def initialize_agent():
    movie_agent_executor = initialize_movie_agent(llm, tools_mix)
    return movie_agent_executor

# Используем session_state для хранения агента и истории чата между запросами
if 'movie_agent_executor' not in st.session_state:
    st.session_state['movie_agent_executor'] = initialize_agent()

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Функция для очистки памяти агента и диалога
def clear_chat():
    st.session_state['chat_history'] = []
    st.session_state['movie_agent_executor'] = initialize_agent()

# Основной заголовок и оформление
st.markdown("<h1 style='text-align: center; color: #FF6347;'>🎬 Movie Search Bot</h1>", unsafe_allow_html=True)

# Примеры запросов в боковой панели
with st.sidebar:
    st.write("### Примеры запросов:")
    st.markdown("""
        - **Поиск фильма по описанию:** "Фильм, где человек после авиакрушения несколько лет выживает на острове"
        - **Поиск описания фильма по названию:** "Расскажи про фильм Титаник"
        - **Рекомендация фильма по жанру или актеру:** "Посоветуй мне фильм в жанре драма"; "Какие фильмы с Леонардо ДиКаприо ты можешь порекомендовать?"
        - **Группировка и поиск используя сложные запросы:** "Топ-5 самых популярных фильмов в жанре триллер"; "Покажи фильмы, в которых снимался Джонни Депп, и отсортируй их по рейтингу"
    """)

# Ввод запроса и кнопки
input_text = st.text_input("Введите запрос", key="input_text", placeholder="Например: Порекомендуй драму с ДиКаприо")

# Стилизация кнопок
col1, col2 = st.columns([1, 1], gap="small")

with col1:
    send_button = st.button("Отправить запрос", key="send_button", use_container_width=True)

with col2:
    clear_button = st.button("Очистить диалог", key="clear_button", use_container_width=True)

# Обработка кнопок с их действиями
if send_button and input_text:
    try:
        with st.spinner("🍿 Агент обрабатывает запрос..."):
            response = get_movie_agent_response(st.session_state['movie_agent_executor'], input_text)
            st.session_state['chat_history'].append((input_text, response))
    except Exception as e:
        st.session_state['chat_history'].append((input_text, "Извините, я не смог обработать ваш запрос"))

if clear_button:
    clear_chat()

# Отображение чата в стиле Telegram с правильным порядком
st.write("### Диалог:")

chat_container = st.container()

with chat_container:
    for query, answer in st.session_state['chat_history']:
        # Сначала сообщение пользователя (справа)
        st.markdown(f"<div style='text-align: right; padding: 10px; background-color: #F0F8FF; border-radius: 10px; margin-bottom: 10px;'>"
                    f"<strong>Вы:</strong> {query}</div>", unsafe_allow_html=True)
        # Затем ответ бота (слева)
        st.markdown(f"<div style='text-align: left; padding: 10px; background-color: #E0FFFF; border-radius: 10px; margin-bottom: 10px;'>"
                    f"<strong>Movie Search Bot:</strong> {answer}</div>", unsafe_allow_html=True)
