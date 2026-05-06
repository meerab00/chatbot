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

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sirf last 10 messages history mein rakho
history = st.session_state.chat_history[-10:]

for msg in history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

user_input = st.chat_input("Kuch poochein...")

if user_input:
    st.chat_message("user").write(user_input)
    chain = get_chain()
    with st.spinner("Soch raha hoon..."):
        try:
            response = chain.invoke({
                "question": user_input,
                "chat_history": history
            })
        except Exception as e:
            st.session_state.chat_history = []
            st.error("Error aaya, chat history clear kar di. Dobara try karo!")
            st.stop()

    st.chat_message("assistant").write(response)
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))
