import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key
)

st.set_page_config(page_title="LangChain Chatbot", page_icon="🤖")
st.title("🤖 LangChain Chatbot")

# Session memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # show user msg
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # get response
    response = llm.invoke(st.session_state.chat_history)

    # show bot msg
    st.chat_message("assistant").write(response.content)
    st.session_state.chat_history.append(AIMessage(content=response.content))
