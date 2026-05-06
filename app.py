import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# LangChain Chain with memory
def get_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=st.secrets["OPENAI_API_KEY"]
    )
    return prompt | llm | StrOutputParser()

# UI
st.title("🦜 LangChain Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

# Input
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
    
    # Update history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))
