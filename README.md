# QnA ChatBot using Langchain 🔗 and Streamlit
A mini project that showcases a QnA-style LLM application built with Streamlit for UI and the Langchain ecosystem for LLM orchestration.


## Index:
- [Project Details](#-project-details)
    - [Aim](#aim)
    - [Features](#features)
    - [Tech Stack](#tech-stack)
- [Steps to run](#-steps-to-run)
- [Future Improvements](#-future-improvements)
- [Contributions](#-contributions)
- [License](#-license)
- [Contact](#-contact)


## 🎯 Project Details:
### Aim:
To provide an interactive chatbot interface that leverages the power of large language models (LLMs), allowing seamless provider switching and real-time response handling.  
It also serves as a way to apply my learnings from the Langchain ecosystem into a practical and useful project.


### Features:

+ 🔄 Switch between multiple LLM providers:
    - OpenAI
    - Groq
    - Ollama (Local LLM)

+ 🧠 Dynamically lists available models based on the selected provider.

+ 🔎 Supports LangSmith tracing for better observability and debugging.

- 🛠️ **Built using LangChain tools:**
  - `ChatPromptTemplate`
  - `ChatMessageHistory` + History Trimmer
  - LangChain Expression Language (LCEL)
  - `RunnablePassthrough`, `RunnableWithMessageHistory`
  - `StringOutputParser`

- 🚧 Change LLM provider and model **on the fly** with zero restarts.


### Tech Stack:
- **Frontend:** Streamlit  
- **LLM Orchestration:** LangChain


## 🚀 Steps to run:

1. **Clone the repository:**
    ```bash
    git clone --depth 1 https://github.com/Bbs1412/QnA_ChatBot
    ```
    
1. **Set up virtual environment:**
    ```bash
    cd QnA_ChatBot
    python -m venv venv
    
    venv\Scripts\activate
    # or
    source venv/bin/activate

    pip install -r requirements.txt
    ```

1. **Set the environment variables *(Optional)*:**
   - Either create a `.env` file in the root directory:
     ```env
     OPENAI_API_KEY=your_openai_api_key
     GROQ_API_KEY=your_groq_api_key
     GEMINI_API_KEY=your_gemini_api_key
     ```

   - Or use Streamlit's config (`.streamlit/secrets.toml`):
     ```toml
     [OpenAI]
     API_KEY = "your_openai_api_key"

     [Groq]
     API_KEY = "your_groq_api_key"

     [Google]
     API_KEY = "your_google_api_key"
     ```
      
1. Run the app:
    ```bash
    streamlit run app.py
    ```


## 📈 Future Improvements:
- Add sliders to let users control `temperature`, `max output tokens` and `context size` dynamically.
- Ensure these parameters are only passed to the LLM when explicitly set by the user.
- This avoids overriding the default values defined by some providers.
- (e.g., certain Ollama models), which differ from standard defaults like `1`.
- Add `Google` provider.

## 🤝 Contributions:
   Any contributions or suggestions are welcome! 


## 📜 License: 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?logo=open-source-initiative)](LICENSE)

- This project is licensed under the `MIT License`.
- See the [LICENSE](LICENSE) file for details.


## 📧 Contact:
- **Email -** [bhushanbsongire@gmail.com](mailto:bhushanbsongire@gmail.com)
