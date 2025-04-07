# Streamlit Chatbot with OpenAI API

from time import sleep
import streamlit as st
from typing import List
from langchain_groq import ChatGroq

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ------------------------------------------------------------------------------
# Globals:
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="Q&A ChatBot",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "init" not in st.session_state:
    st.session_state.api = None
    st.session_state.providers = ["OpenAI", "Groq", "Ollama"]
    st.session_state.model = None

    st.session_state.chat_history = [
        AIMessage("Hello 👋! How can I assist you today?")]

    st.session_state.init = True


# ------------------------------------------------------------------------------
# Helpers:
# ------------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_model_list(provider: str) -> List[str]:
    """Returns the list of available models from given provider."""
    sleep(3)

    if provider == "Groq":
        from groq import Groq
        client = Groq(api_key=st.secrets.Groq.API_KEY)
        return [model.id for model in client.models.list().data]

    elif provider == "OpenAI":
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets.OpenAI.API_KEY)
        return [model.id for model in client.models.list().data]

    elif provider == "Ollama":
        import ollama
        return [model.model for model in ollama.list().models]

    else:
        return ["demo-model"]


# ------------------------------------------------------------------------------
# Langchain things:
# ------------------------------------------------------------------------------

chat_history = ChatMessageHistory()
template = ChatPromptTemplate.from_messages(
    messages=[
        ("system", "You are a helpful assistant {llm_name} who responds to questions in not more than 20 sentences. You can use markdown and code blocks to format your answers. You can also use emojis to make your answers more engaging. Please be concise and clear in your responses."),
        MessagesPlaceholder(variable_name="chat_history"),
    ]
)

llm = None

if st.session_state.provider == "OpenAI":
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=st.session_state.model, api_key=st.secrets.OpenAI.API_KEY)

elif st.session_state.provider == "Groq":
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model=st.session_state.model, api_key=st.secrets.Groq.API_KEY)

elif st.session_state.provider == "Ollama":
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=st.session_state.model)

else:
    st.error("Please select some model to continue...", icon="🤖")
    
# llm =
# chain =


# ------------------------------------------------------------------------------
# Sidebar:
# ------------------------------------------------------------------------------

st.sidebar.title("Settings ⚙️")

st.sidebar.selectbox(
    options=st.session_state.providers,
    label="Select Provider:",
    index=None,
    key="provider"
)

st.sidebar.selectbox(
    label="Select Model:",
    options=get_model_list(st.session_state.provider),
    index=None,
    placeholder="Choose Model" if st.session_state.provider else "Choose provider first",
    key="model"
)

st.sidebar.text_input(
    label="Enter your API Key:",
    placeholder="API Key 👀",
    type="password",
    key="api_key"
)

st.sidebar.text_input(
    label="AI Name:",
    value="Sahayak",
    max_chars=50
)

# ------------------------------------------------------------------------------
# Main page content:
# ------------------------------------------------------------------------------

st.header(":orange[Q&A ChatBot] 💬 with :blue[LangChain] 🔗", divider='red')

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        st.chat_message("user").markdown(message.content)
    elif isinstance(message, AIMessage):
        st.chat_message("assistant").markdown(message.content)
    # else:
    #     st.chat_message("system").markdown(message.content)


if user_message := st.chat_input(
        placeholder="Ask me anything! 🤔", accept_file=False):
    st.session_state.chat_history.append(HumanMessage(user_message))
    st.chat_message("user").markdown(user_message)

    # get ai response:
