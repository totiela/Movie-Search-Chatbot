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

langchain.debug=False

@st.cache_resource
def setup():
    download_and_prepare()
    llm, embedder = initialize_llm()
    ensemble_retriever = initialize_retrievers(embedder, "faiss_mistral-7b-v2_embed_index", "retrievers\\bm25_retriever.pkl")
    chain = initialize_prompts_and_chain(llm, ensemble_retriever)
    agent_executor_sql = initialize_sql_agent("databases\movies_with_descriptions.db", llm)
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

# Устанавливаем стили, чтобы увеличить отступы и сделать интерфейс более просторным
st.markdown(
    """
    <style>
    .example-column {
        padding-right: 30px; /* Отступ между колонками */
    }
    .chat-column {
        padding-left: 30px;
    }
    .stTextInput > div > input {
        width: 100%;
    }
    .stButton > button {
        width: 100%;
    }
    .dialogue {
        background-color: #f4f4f4;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Лэйаут с двумя колонками: слева примеры запросов, справа чат
col1, col2 = st.columns([2, 4])  # Изменяем пропорции колонок, чтобы увеличить левую колонку

# Левая колонка с примерами запросов
with col1:
    st.write("### Примеры запросов", unsafe_allow_html=True)
    st.markdown("""
    <div class="example-column">
        <ul>
            <li><b>Поиск фильма по описанию:</b><br>"Фильм, где человек после авиакрушения несколько лет выживает на острове"</li>
            <li><b>Поиск описания фильма по названию:</b><br>"Расскажи про фильм Титаник"</li>
            <li><b>Рекомендация фильма по жанру или актеру:</b><br>"Посоветуй мне фильм в жанре драма";<br>"Какие фильмы с Леонардо ДиКаприо ты можешь порекомендовать?"</li>
            <li><b>Группировка и поиск используя сложные запросы:</b><br>"Топ-5 самых популярных фильмов в жанре триллер";<br>"Покажи фильмы, в которых снимался Джонни Депп, и отсортируй их по рейтингу"</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Правая колонка с диалогом
with col2:
    st.title('Movie Search Bot')

    # Ввод запроса
    input_text = st.text_input("Введите запрос")

    # Кнопки для отправки запроса и очистки чата
    with st.container():
        col_button1, col_button2 = st.columns([1, 1])  # Создаем две колонки для кнопок

        # Кнопка отправки запроса
        with col_button1:
            if st.button("Отправить запрос"):
                if input_text:
                    # Показываем индикатор загрузки
                    with st.spinner("Агент обрабатывает запрос..."):
                        response = get_movie_agent_response(st.session_state['movie_agent_executor'], input_text)
                        st.session_state['chat_history'].append((input_text, response))

        # Кнопка для очистки диалога и памяти агента
        with col_button2:
            if st.button("Очистить диалог"):
                clear_chat()

    # Визуальное отображение последних 3 сообщений как переписки
    st.write("### Диалог:")

    # Прокрутка через history (последние 3 сообщения показываем)
    if st.session_state['chat_history']:
        for query, answer in st.session_state['chat_history'][-3:]:
            st.markdown(f"<div class='dialogue'><b>Вы:</b> {query}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='dialogue'><b>Movie Search Bot:</b> {answer}</div>", unsafe_allow_html=True)

    # Если сообщений больше, даём возможность прокрутки через history
    if len(st.session_state['chat_history']) > 3:
        with st.expander("Показать всю историю переписки"):
            for query, answer in st.session_state['chat_history']:
                st.markdown(f"<div class='dialogue'><b>Вы:</b> {query}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='dialogue'><b>Movie Search Bot:</b> {answer}</div>", unsafe_allow_html=True)