from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

def initialize_sql_agent(db_path, llm):
    # Подключение к базе данных SQLite
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    
    # Создание SQL toolkit для работы с агентом
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Создание SQL-агента с настройками
    agent_executor_sql = create_sql_agent(
        llm=llm, toolkit=toolkit, verbose=True, handle_parsing_errors=True)
    
    return agent_executor_sql
