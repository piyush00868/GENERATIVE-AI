import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

model = ChatMistralAI(model="mistral-small-latest")

st.set_page_config(
    page_title="Mistral AI Chat",
    page_icon="🤖"
)

st.title("🤖 Chat with Mistral AI")

# ---------------- MODE SELECTION ----------------

st.sidebar.title("Choose AI Mode")

choice = st.sidebar.radio(
    "Select a personality:",
    [
        "😡 Angry Mode",
        "😐 Normal Mode",
        "😂 Fun Mode",
        "😢 Sad Mode"
    ]
)

if choice == "😡 Angry Mode":
    mode = "You are an angry AI model. You respond to the user in a very angry tone."

elif choice == "😐 Normal Mode":
    mode = "You are a normal AI model. You respond to the user in a neutral tone."

elif choice == "😂 Fun Mode":
    mode = "You are a fun AI model. You respond to the user in a playful tone."

elif choice == "😢 Sad Mode":
    mode = "You are a sad AI model. You respond to the user in a melancholic tone."


# ---------------- MESSAGE HISTORY ----------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=mode)
    ]


# ---------------- DISPLAY CHAT ----------------

for message in st.session_state.messages:

    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)


# ---------------- CHAT INPUT ----------------

prompt = st.chat_input("Type your message...")

if prompt:

    # User message
    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    with st.chat_message("user"):
        st.write(prompt)

    # AI response
    response = model.invoke(
        st.session_state.messages
    )

    st.session_state.messages.append(
        AIMessage(content=response.text)
    )

    with st.chat_message("assistant"):
        st.write(response.text)