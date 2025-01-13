import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
import certifi

## Arxiv and wikipedia Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

search=DuckDuckGoSearchRun(name="Search") # inbuilt tool

load_dotenv()

os.environ['SSL_CERT_FILE'] = certifi.where()

st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Set default API key if none provided
if not api_key:
    api_key = "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"

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi,I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="mixtral-8x7b-32768",
        streaming=True
    )
    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, # it gives output without relying on the chat history, that is on the basis of the current input
        handle_parsing_errors=True # handle_parsing_errors=True :- If the LLM outputs an invalid format, so instead of crashing, the agent will Catch the parsing error & will return the raw LLM output as a final answer. This acts as a fallback mechanism
    )


    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True) # StreamlitCallbackHandler is showing the actions of an agent in streamlit app
        response = search_agent.run(prompt)
        st.session_state.messages.append({'role': 'assistant', "content": response})
        st.write(response)
