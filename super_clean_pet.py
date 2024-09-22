# Импорт библиотек
from dotenv import load_dotenv
import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit 
from getpass import getpass
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import Tool, AgentExecutor, create_react_agent
import pandas as pd
from langchain import hub
import pickle

def initialize_llm(model_name='google/gemma-2-27b-it'):
    # Загружаем переменные окружения из .env
    load_dotenv()  # Это загрузит все переменные из .env файла
    
    # Получаем API ключ из переменных окружения
    api_key = os.getenv('NVIDIA_API_KEY')

    if not api_key:
        raise ValueError("API ключ не найден! Проверьте файл .env")

    # Инициализируем LLM ChatNVIDIA
    llm = ChatNVIDIA(
        model=model_name,
        nvidia_api_key=api_key
    )

    # Инициализация встраивания
    embedder = NVIDIAEmbeddings(
        model='nvidia/nv-embedqa-mistral-7b-v2',
        api_key=api_key
    )

    return llm, embedder

llm, embedder = initialize_llm()

def initialize_retrievers(faiss_path, bm25_path, weights=[0.2, 0.8]):
    # Загрузка модели FAISS
    embeddings = embedder  # предполагается, что embedder определен где-то ранее
    db_embed = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db_embed.as_retriever()

    # Загрузка модели BM25
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)

    # Инициализация EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25, retriever],  # список ретриверов
        weights=weights  # веса для каждого ретривера
    )

    return ensemble_retriever

ensemble_retriever = initialize_retrievers("faiss_mistral-7b-v2_embed_index", "bm25_retriever.pkl")

def initialize_prompts_and_chain(llm, retriever):
    # Шаблон для извлечения названия фильма
    film_name_template = """
    From the text below find the name of the movie and return it. If you dont find the name of the movie - return key words form the text.
    DON'T invent it yourself, use only the text below!!! Also dont answer the question given, just follow my instructions.
    Return it in Russian if possible.

    Example: 'Какой рейтинг у Индиана джонса?'
    Return: 'Индиана Джонс'

    Example: 'сколько лет главному герою джумандлжи'
    Return: Джуманджи

    Example: 'драмма написанная тарантино в 1998 году'
    Return: драмма Тарантино 1998 год

    Example: 'фильм где много экшена, драки, там еще играет Ди Каприо'
    Return: 'фильм экшен драки Ди Каприо'.

    Example: 'фильм, где человек после авиакрушения несколько лет выживает на острове'
    Return: 'фильм авиакрушение выживать остров'

    Example: 'Психически больной человек у которого много личностей'
    Return: 'Психически больной человек много личностей'

    text: {context}
    """

    # Шаблон для ответов
    answer_template = """
    Answer to the question or statement only on the following context.

    1. You are to provide clear, concise, and direct responses.
    2. Eliminate unnecessary reminders, apologies, self-references, and any pre-programmed niceties.
    3. Maintain a casual tone in your communication.
    4. Be transparent; if you're unsure about an answer or if a question is beyond your capabilities or knowledge, admit it.
    5. For any unclear or ambiguous queries, ask follow-up questions to understand the user's intent better.
    6. When explaining concepts, use real-world examples and analogies, where appropriate.
    7. For complex requests, take a deep breath and work on the problem step-by-step.
    8. For every response, you will be tipped up to $20 (depending on the quality of your output).

    It is very important that you get this right. Multiple lives are at stake.
    DON'T use your knowledge and DON'T invent it yourself, use only the text below!!!:

    context: {context}

    Question: {question}

    For movie in asnwer also use year 'Дата релиза' in '()' after the movie title, rating of the movie and description of the movie if possible.
    Ответь на русском языке.
    Если не знаешь ответ ответь что ты не знаешь.
    """

    # Создаём промпты из шаблонов
    film_name_prompt = ChatPromptTemplate.from_template(film_name_template)
    answer_prompt = ChatPromptTemplate.from_template(answer_template)

    # Функция для формирования строки из полученных документов
    def format_docs(docs):
        return "\n\n".join([f"Content: {d.page_content}\nMetadata: {d.metadata}" for d in docs])

    # Создание цепочки
    chain = (
        {"context": ({'context': RunnablePassthrough()} | film_name_prompt | llm | StrOutputParser()) | retriever | format_docs, "question": RunnablePassthrough()}
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    return chain


chain = initialize_prompts_and_chain(llm, ensemble_retriever)

def initialize_sql_agent(db_path, llm=llm):
    # Подключение к базе данных SQLite
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

    # Создаем SQLDatabaseToolkit для взаимодействия с базой данных
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Создаем SQL агента с включенным параметром handle_parsing_errors
    agent_executor_sql = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        handle_parsing_errors=True,
    )

    return agent_executor_sql

db_path = "movies_with_descriptions.db"
agent_executor_sql = initialize_sql_agent(db_path)

def create_tools(chain, agent_executor_sql):
    tools_mix = [
        Tool(
            name='RetrieverAgent',
            func=chain.invoke,
            description='''Useful to search movie by description and description by movie.
            Also useful for finding similar movies.
            '''
        ),
        Tool(
            name="SQLAgent",
            func=agent_executor_sql.invoke,
            description="""Useful to query sql tables, search and sort movies by genre, actors, directors, movie rating.
            Useful when it is expected a several movies output.
            (there are different tables for genres, actors also key-to-key tables like movie_actors(to compare authors and movies) and movie_genres(to compare movies and genres) that you can use to join data).
            Genres are all in russian with lowercase letters.
            Also useful to search movie if RetrieverAgent didn't help""",
        )
    ]

    return tools_mix

tools_mix = create_tools(chain, agent_executor_sql)



# prompt = hub.pull("hwchase17/react-chat")
prompt = hub.pull("sanchezzz/russian_react_chat")
# prompt = hub.pull("sanchezzz/russian_movie_react_chat")

from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=2, memory_key='chat_history')
agent = create_react_agent(llm, tools_mix, prompt)

movie_agent_executor_memory = AgentExecutor(agent=agent, tools=tools_mix, verbose=True, memory=memory, max_iterations=8, handle_parsing_errors=True)

def initialize_movie_agent(llm=llm, tools_mix=tools_mix, prompt_name='sanchezzz/russian_react_chat', memory_k=2, max_iterations=8):
    # Получаем промпт с хаба
    prompt = hub.pull(prompt_name)

    # Инициализируем память с указанным окном
    memory = ConversationBufferWindowMemory(k=memory_k, memory_key='chat_history')

    # Создаем агента с использованием LLM, инструментов и промпта
    agent = create_react_agent(llm, tools_mix, prompt)

    # Создаем агент с исполнителем, памятью и обработкой ошибок
    movie_agent_executor_memory = AgentExecutor(
        agent=agent,
        tools=tools_mix,
        verbose=True,
        memory=memory,
        max_iterations=max_iterations,
        handle_parsing_errors=True
    )

    return movie_agent_executor_memory

movie_agent_executor = initialize_movie_agent()

def get_movie_agent_response(movie_agent_executor, input_text):
    # Ввод текста для агента
    response = movie_agent_executor.invoke({"input": input_text})

    # Возвращаем результат
    return response['output']

langchain_debug=True

response = get_movie_agent_response(movie_agent_executor, 'топ-5 самых лучших фильма в которых снимался ди каприо, выведи их рейтинги и описания')
print(response)