# Streamlit Chatbot with OpenAI API

import os
from time import sleep
import streamlit as st
from typing import List
from dotenv import load_dotenv

from typing import Generator
from operator import itemgetter
from langchain_core.messages import trim_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough

load_dotenv()

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
    st.session_state.providers = ["OpenAI", "Groq", "Ollama", "Google"]
    st.session_state.provider = None
    st.session_state.model = None
    st.session_state.user_api_key = None

    st.session_state.chat_history = [
        AIMessage("Hello 👋! How can I assist you today?")]

    st.session_state.init = True


# ------------------------------------------------------------------------------
# Helpers:
# ------------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_model_list(provider: str) -> List[str]:
    """Returns the list of available models from given provider."""
    # sleep(3)

    # Update API key in session state:
    st.session_state.user_api_key = get_api_key(provider)

    if provider == "Groq":
        from groq import Groq
        client = Groq(api_key=st.session_state.user_api_key)
        return [model.id for model in client.models.list().data]

    elif provider == "OpenAI":
        from openai import OpenAI
        client = OpenAI(api_key=st.session_state.user_api_key)
        return [model.id for model in client.models.list().data]

    elif provider == "Ollama":
        import ollama
        return [model.model for model in ollama.list().models]

    elif provider == "Google":
        from google import genai
        client = genai.Client(api_key=st.session_state.user_api_key)
        return [model.name for model in client.models.list()]

    else:
        return ["demo-model"]


def get_api_key(provider: str) -> str | None:
    """Returns the API key, if present in secrets or .env"""
    if provider == "OpenAI":
        try:
            return st.secrets.OpenAI.API_KEY
        except:
            pass
        try:
            return os.getenv("OPENAI_API_KEY")
        except:
            pass

    elif provider == "Groq":
        try:
            return st.secrets.Groq.API_KEY
        except:
            pass
        try:
            return os.getenv("GROQ_API_KEY")
        except:
            pass

    elif provider == "Google":
        try:
            return st.secrets.Google.API_KEY
        except:
            pass
        try:
            return os.getenv("GEMINI_API_KEY")
        except:
            pass

    return None


def get_session_history() -> BaseChatMessageHistory:
    if "lc_chat_hist" not in st.session_state:
        st.session_state.lc_chat_hist = ChatMessageHistory()
    return st.session_state.lc_chat_hist


# ------------------------------------------------------------------------------
# Langchain things:
# ------------------------------------------------------------------------------

def get_llm_response_stream(prompt: str) -> Generator[str, None, None]:
    """Returns the response from LLM for given prompt using Generator."""
    # Chat Prompt Template:
    template = ChatPromptTemplate.from_messages(
        messages=[
            ("system", "You are a helpful assistant '{llm_name}' who responds to questions in not more than 20 sentences. You can use markdown and code blocks to format your answers. You can also use emojis to make your answers more engaging. Please be concise and clear in your responses."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{new_input}")
        ]
    )

    # Ensure model and provider are selected:
    if not st.session_state.provider:
        raise ValueError(
            "Provider not selected, please select a provider first.")

    if not st.session_state.model:
        raise ValueError("Model not selected, please select a model first.")

    # Set-up LLM:
    llm = None

    if st.session_state.provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=st.session_state.model, api_key=st.session_state.user_api_key)

    elif st.session_state.provider == "Groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model=st.session_state.model, api_key=st.session_state.user_api_key)

    elif st.session_state.provider == "Ollama":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=st.session_state.model)
        # llm = ChatOllama(base_url="http://host.docker.internal:11434" ,model=st.session_state.model)

    elif st.session_state.provider == "Google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=st.session_state.model, api_key=st.session_state.user_api_key)

    else:
        st.error("Some un-expected error occurred...", icon="🤖")

    # Output parser:
    parser = StrOutputParser()

    # Chain with trimmer:
    # Trimmer:
    trimmer = trim_messages(
        max_tokens=2000, strategy="last",
        token_counter=llm, include_system=False,
        allow_partial=True, start_on=HumanMessage
    )

    # Chain them all:
    chain = (
        # Set "messages" key equal to chat_history
        RunnablePassthrough.assign(
            messages=itemgetter("chat_history") | trimmer)
        # Set "chat_history" key equal to "messages" (default output key of trimmer)
        | RunnablePassthrough.assign(chat_history=itemgetter("messages"))
        | template
        | llm
        | parser
    )
    # Tested and WORKING 🥳

    # # Chain without the trimmer:
    # # Comment out the trimmer and chain above to use this:
    # chain = (
    #     template
    #     | llm
    #     | parser
    # )

    llm_with_history = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_session_history,
        input_messages_key="new_input",
        history_messages_key="chat_history",
    )

    # Run the chain (streaming):
    yield from llm_with_history.stream(
        input={
            "new_input": prompt,
            "llm_name": st.session_state.name,
        }
    )


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
    key="user_api_key",
    value=st.session_state.user_api_key if st.session_state.user_api_key else None,
)

if st.sidebar.text_input(label="AI Name:", value="Sahayak", max_chars=50, key="name"):
    st.session_state.chat_history[0] = AIMessage(
        f"Hello 👋 I'm {st.session_state.name}! How can I assist you today?")

if st.sidebar.button("Reset Chat History", key="reset"):
    st.session_state.chat_history = st.session_state.chat_history[:1]
    st.session_state.lc_chat_hist = ChatMessageHistory()
    st.rerun()


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

    # Get AI response:
    full = ""
    trail_char = "█"  # "█", "▌", "|", "•"

    with st.chat_message("assistant"):
        with st.spinner("Generating Response..."):
            message_placeholder = st.empty()

            for chunk in get_llm_response_stream(user_message):
                full += chunk
                message_placeholder.container(
                    border=True).markdown(full + trail_char)

    st.session_state.chat_history.append(AIMessage(full))
    st.rerun()
