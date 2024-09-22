from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain import hub

def initialize_movie_agent(llm, tools_mix, prompt_name='sanchezzz/russian_react_chat', memory_k=2, max_iterations=8):
    prompt = hub.pull(prompt_name)
    memory = ConversationBufferWindowMemory(k=memory_k, memory_key='chat_history')
    agent = create_react_agent(llm, tools_mix, prompt)

    movie_agent_executor_memory = AgentExecutor(
        agent=agent, tools=tools_mix, verbose=True, memory=memory, 
        max_iterations=max_iterations, handle_parsing_errors=True)
    
    return movie_agent_executor_memory

def get_movie_agent_response(movie_agent_executor, input_text):
    response = movie_agent_executor.invoke({"input": input_text})
    return response['output']
