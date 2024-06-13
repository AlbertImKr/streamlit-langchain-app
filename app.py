import os

import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent, load_tools
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory

def create_agent_chain(history):
    chat = ChatOpenAI(
        model_name = os.environ["OPENAI_API_MODEL"],
        temperature = os.environ["OPENAI_API_TEMPERATURE"],
    )
    
    tools = load_tools(["ddg-search","wikipedia"])
    
    prompt = hub.pull("hwchase17/openai-tools-agent")
    
    # OpenAi Functions Agent가 사용할 수 있는 설정으로 Memory 초기화
    memory = ConversationBufferMemory(
        chat_memory=history, memory_key="chat_history",return_messages=True
    )
    
    agent = create_openai_tools_agent(chat, tools, prompt)
    return AgentExecutor(agent=agent,tools=tools,memory=memory)

load_dotenv()

st.title("langchain-streamlit-app")

history = StreamlitChatMessageHistory()

for message in history.messages:
    st.chat_message(message.type).write(message.content)

prompt = st.chat_input("what is up?")

if prompt: # 입력한 문자열이 있는 경우(None도 아니고 빈 문자열도 아닌 경우)
    with st.chat_message("user"): # 사용자의 아이콘으로
        st.markdown(prompt) # prompt를 마크다운으로 정형화해 표시
    
    with st.chat_message("assistant"): # AI의 아이콘으로
        callback = StreamlitCallbackHandler(st.container())
        agent_chain = create_agent_chain(history)
        response = agent_chain.invoke(
            {"input": prompt},
            {"callbacks":[callback]},
        )
        
        st.markdown(response) # 응답을 마크다운으로 정형화해 표시