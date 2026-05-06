import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

def get_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    llm = ChatGroq(
        model="llama3-8b-8192",
        api_key=st.secrets["GROQ_API_KEY"]
    )
    return prompt | llm | StrOutputParser()

st.title("🦜 LangChain Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

user_input = st.chat_input("Kuch poochein...")

if user_input:
    st.chat_message("user").write(user_input)
    chain = get_chain()
    with st.spinner("Soch raha hoon..."):
        response = chain.invoke({
            "question": user_input,
            "chat_history": st.session_state.chat_history
        })
    st.chat_message("assistant").write(response)
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))
