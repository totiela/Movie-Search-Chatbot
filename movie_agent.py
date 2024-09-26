from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain import hub

# Инициализация агента для работы с фильмами
def initialize_movie_agent(llm, tools_mix, prompt_name='sanchezzz/russian_react_chat', memory_k=2, max_iterations=8):
    # Загрузка кастомного промпта из Hub (в данном случае для русскоязычного агента)
    prompt = hub.pull(prompt_name)

    # Настройка памяти агента: ConversationBufferWindowMemory сохраняет последние 'k' сообщений
    memory = ConversationBufferWindowMemory(k=memory_k, memory_key='chat_history')

    # Создаем агента с реактивной стратегией, которая будет реагировать на ввод пользователя с использованием инструментов
    agent = create_react_agent(llm, tools_mix, prompt)

    # Экзекутор агента: оборачиваем агента в Executor, который обрабатывает его взаимодействие с инструментами и памятью
    movie_agent_executor_memory = AgentExecutor(
        agent=agent, tools=tools_mix, verbose=True,  # Включаем подробный вывод для отслеживания работы агента
        memory=memory,  # Используем память, чтобы учитывать предыдущие запросы
        max_iterations=max_iterations,  # Максимальное количество шагов, которое агент может выполнить за один запрос
        handle_parsing_errors=True  # Обработка ошибок парсинга ввода пользователя
    )
    
    return movie_agent_executor_memory  # Возвращаем агента с памятью и инструментами

# Функция для получения ответа от агента на основе введенного текста
def get_movie_agent_response(movie_agent_executor, input_text):
    # Передаем ввод пользователя агенту и получаем ответ
    response = movie_agent_executor.invoke({"input": input_text})
    
    # Возвращаем непосредственно текст ответа
    return response['output']
