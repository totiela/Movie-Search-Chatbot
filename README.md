# 🎬 Чат-Бот для Поиска Фильмов

[![Запустить приложение](https://img.shields.io/badge/Streamlit-Запустить%20приложение-red?style=for-the-badge&logo=streamlit)](https://movie-search-chatbot-by-sanchezzz.streamlit.app/)

## 📜 Описание проекта

**Movie Search Chatbot** — это умный чат-бот, который делает поиск фильмов простым, увлекательным и эффективным. Забудьте о скучных поисковых системах — просто опишите сюжет, вспомните актёра или режиссёра, и бот мгновенно предложит вам точные результаты. Хотите узнать больше о любимом фильме, найти похожие картины или провести сложный анализ по актёрскому составу? Этот бот — ваш персональный гид в мире кино. Будь то редкие фильмы, культовые классики или современные хиты, этот бот поможет найти нужное в пару кликов.


### Целевая аудитория:
- **Киноэнтузиасты**, которым нужны персонализированные рекомендации на основе предпочтений.
- Пользователи, помнящие **сюжетные фрагменты**, имена актёров или режиссёра, но забывшие название фильма.
- Люди, которым нужна **сложная агрегация информации** о фильмах, недоступная в публичных источниках либо проблематичная для вычисления.
- **Люди с ограниченным временем**, которым нужен быстрый и точный поиск информации о конкретных фильмах.
  
## 🚀 Ключевые возможности

1. **Поиск по описанию фильма:** Чат-бот может находить фильмы по ключевым словам или описанию, которые вы помните.
2. **Рекомендации похожих фильмов:** На основе введённого описания бот предложит фильмы, которые могут вам понравиться.
3. **Поиск информации о фильмах** По введённому названию фильма пожно запросить любую интерсующую информацию.
4. **Анализ сложных запросов:** Например, средний рейтинг фильмов, где играли несколько определенных актёров вместе.
5. **Интерактивный чат с историей:** Вы можете взаимодействовать с ботом через диалог, сохраняя историю общения.
6. **Общение на любые темы:** Хоть и прямая задача бота заключается в поиске фильмов, это не отменяет возможности **пообщаться** с ним **о чем угодно**.
7. **Использование мощных языковых моделей:** Под капотом работает модель **gemma2-27b** от Google через API NVIDIA.
8. **Объединение поисковых методов:** Система использует как **BM25**, так и векторный поиск с помощью **FAISS** для улучшения точности поиска фильмов.

## 🛠 Особенности реализации

1. **Языковая модель:** Чат-бот использует модель **gemma2-27b**, которая обеспечивает глубокое понимание текста и контекста, предлагая более осмысленные ответы на запросы пользователей.
2. **Многоагентная архитектура:** Проект реализует два специализированных агента:
   - **SQL-агент** отвечает за сложные SQL-запросы к базе данных фильмов, позволяя искать по актёрам, режиссёрам, жанрам и другим критериям.
   - **Retriever-агент** использует комбинированные поисковые методы (BM25 и FAISS) для нахождения фильмов на основе описания или ключевых слов.
3. **Кастомный ReAct-суперагент:** Управляет взаимодействием с пользователем, координируя работу SQL и Retriever агентов, а также контролирует обработку запросов на русском языке.
4. **Диалог с памятью:** Реализованная память с использованием **ConversationBufferMemory** обеспечивает контекстное взаимодействие, поддерживая непрерывность диалога и запоминая информацию из предыдущих сообщений.
5. **Парсинг данных:** Произведён парсинг данных с портала **IMDb**, где было спарсено более 5000 фильмов. Этими данными были заполнены как SQL-база, так и векторная база FAISS, что позволяет агентам эффективно доставать информацию о фильмах в полном объеме.

## 📚 Необходимые библиотеки и ресурсы

Для работы чат-бота требуются следующие библиотеки и ресурсы:

- **LangChain** — основная библиотека для работы с агентами и создания кастомных цепочек обработки запросов.
- **FAISS** — фреймворк для векторного поиска по описанию и ключевым словам.
- **BM25** — метод обратного индексирования для быстрого поиска по текстам.
- **SQLite** — база данных для хранения структурированных данных о фильмах.
- **Streamlit** — используется для деплоя и создания веб-интерфейса приложения.
- **NVIDIA API** — для работы с моделью gemma2-27b и генерацией естественных ответов на запросы.

(Полный список библиотек указан в [**requirements.txt**](https://github.com/totiela/Movie-Search-Chatbot/blob/main/requirements.txt))

## 🔧 Технические аспекты

1. **Архитектура агентов:**
   - **SQLAgent** выполняет SQL-запросы к базе данных, предоставляя доступ к информации о фильмах, режиссёрах и актёрах.
   - **RetrieverAgent** работает с описаниями фильмов, используя как BM25, так и FAISS для более точных результатов поиска.
2. **Интеграция с LangChain:** 
   - Используются кастомные промпты для агентов, улучшая взаимодействие с пользователем на русском языке.
   - **ReAct агент** управляет выбором действий между агентами, координируя работу Retriever и SQL-агентов.
3. **Память чата:** 
   - **ConversationBufferMemory** позволяет боту запоминать предыдущие вопросы, поддерживая связный диалог с пользователем.
4. **Работа с базами данных:**
   - **FAISS** отвечает за векторный поиск фильмов, анализируя ключевые слова и их векторные представления.
   - **SQLite** поддерживает сложные запросы, такие как поиск фильмов по нескольким актёрам или режиссёрам, анализ жанров и рейтинг фильмов.

## 📦 Деплой

Чат-бот деплоирован на платформе **Streamlit**, что позволило быстро и удобно развернуть веб-приложение для взаимодействия с пользователями. Веб-страница интуитивно понятна и обеспечивает удобный интерфейс для поиска фильмов, рекомендаций и анализа.

Запуск MVP версии осуществляется [по этой ссылке](https://movie-search-chatbot-by-sanchezzz.streamlit.app/).

## 🔮 Планы на будущее

1. Улучшение обработки ошибок.
2. Расширение базы данных фильмов.
3. Добавление новых возможностей, таких как:
   - Поддержка Telegram-бота.
   - Увеличение числа агентов и улучшение их взаимодействия.
   - Оптимизация модели для ускорения работы.

### Описание клонирования репозитория

1. **Клонирование репозитория:**
   Чтобы склонировать репозиторий на локальную машину, выполните следующую команду в терминале:
   ```bash
   git clone https://github.com/totiela/Movie-Search-Chatbot.git
   ```
2. **Переход в директорию проекта**: После клонирования переместитесь в директорию репозитория:
   ```bash
   cd Movie-Search-Chatbot
   ```
3. **Установка зависимостей**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Настройка ключей API и переменных окружения** добавьте их в файл .env
   ```bash
   NVIDIA_API_KEY=your_key_here
   ```
5. **Запуск приложения**
    ```bash
   streamlit run main.py
   ```

### Описание файлов репозитория

- [**.devcontainer/**](https://github.com/totiela/Movie-Search-Chatbot/tree/main/.devcontainer) — папка для настройки контейнера разработчика, помогает запускать проект в изолированной среде.

- [**databases/**](https://github.com/totiela/Movie-Search-Chatbot/tree/main/databases) — папка с базой данных для работы проекта, хранит необходимые данные о фильмах.

- [**faiss_mistral-7b-v2_embed_index/**](https://github.com/totiela/Movie-Search-Chatbot/tree/main/faiss_mistral-7b-v2_embed_index) — содержит индекс FAISS для векторного поиска по описаниям фильмов.

- [**retrievers/**](https://github.com/totiela/Movie-Search-Chatbot/tree/main/retrievers) — папка с ретриверами, используемыми для поиска фильмов по описаниям и ключевым словам.

- [**.gitignore**](https://github.com/totiela/Movie-Search-Chatbot/blob/main/.gitignore) — файл, определяющий, какие файлы и папки следует игнорировать в Git.

- [**README.md**](https://github.com/totiela/Movie-Search-Chatbot/blob/main/README.md) — основной файл с описанием проекта.

- [**download_db.py**](https://github.com/totiela/Movie-Search-Chatbot/blob/main/download_db.py) — скрипт для загрузки необходимых файлов базы данных и индексов.

- [**llm_utils.py**](https://github.com/totiela/Movie-Search-Chatbot/blob/main/llm_utils.py) — вспомогательные функции для работы с LLM (Large Language Models).

- [**main.py**](https://github.com/totiela/Movie-Search-Chatbot/blob/main/main.py) — основной файл для запуска проекта.

- [**movie_agent.py**](https://github.com/totiela/Movie-Search-Chatbot/blob/main/movie_agent.py) — агент для работы с фильмами, использующий инструменты и ретриверы.

- [**prompts_and_chain.py**](https://github.com/totiela/Movie-Search-Chatbot/blob/main/prompts_and_chain.py) — логика работы с промптами и цепочками для обработки запросов.

- [**requirements.txt**](https://github.com/totiela/Movie-Search-Chatbot/blob/main/requirements.txt) — список зависимостей для установки и работы проекта.

- [**retrievers.py**](https://github.com/totiela/Movie-Search-Chatbot/blob/main/retrievers.py) — модуль с настройками ретриверов для поиска фильмов по ключевым словам и описаниям.

- [**sql_agent.py**](https://github.com/totiela/Movie-Search-Chatbot/blob/main/sql_agent.py) — SQL агент для взаимодействия с базой данных фильмов.

- [**tools.py**](https://github.com/totiela/Movie-Search-Chatbot/blob/main/tools.py) — инструменты, используемые в проекте, такие как агенты и ретриверы.
