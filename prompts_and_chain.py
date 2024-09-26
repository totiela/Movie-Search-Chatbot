from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Функция для инициализации шаблонов промптов и создания цепочки с LLM и ретривером
def initialize_prompts_and_chain(llm, retriever):
    # Шаблон для извлечения названия фильма из текста
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

    # Шаблон для ответа на запрос пользователя, использующего контекст
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

    For movie in answer also use year 'Дата релиза' in '()' after the movie title, rating of the movie and description of the movie if possible.
    Ответь на русском языке.
    Если не знаешь ответ, ответь что ты не знаешь.
    """

    # Создаём промпты из шаблонов
    film_name_prompt = ChatPromptTemplate.from_template(film_name_template)  # Промпт для поиска названия фильма
    answer_prompt = ChatPromptTemplate.from_template(answer_template)  # Промпт для формирования ответа

    # Функция для форматирования документов в строку
    def format_docs(docs):
        # Форматируем документы с контентом и метаданными
        return "\n\n".join([f"Content: {d.page_content}\nMetadata: {d.metadata}" for d in docs])

    # Создание цепочки:
    # - Извлекаем контекст для поиска названия фильма
    # - Передаём в ретривер для поиска фильмов по ключевым словам
    # - Затем создаём ответ с помощью промпта и LLM
    chain = (
        {"context": ({'context': RunnablePassthrough()} | film_name_prompt | llm | StrOutputParser())  # Извлечение контекста и поиск названия
        | retriever | format_docs,  # Поиск фильмов через ретривер по ключевым словам
        "question": RunnablePassthrough()}  # Вопрос пользователя передаётся напрямую в цепочку
        | answer_prompt  # Формируем ответ на основе контекста и вопроса
        | llm  # Используем LLM для генерации ответа
        | StrOutputParser()  # Парсим результат
    )

    return chain  # Возвращаем финальную цепочку
