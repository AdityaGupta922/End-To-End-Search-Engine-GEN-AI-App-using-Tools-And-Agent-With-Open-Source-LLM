import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
import certifi

# Arxiv and wikipedia Tools
# Used the inbuilt tool of wikipedia & Arxiv
# This creates a tool that can search Wikipedia & Arxiv. The parameters specify:
# - top_k_results=1: Return only the top result
# - doc_content_chars_max=250: Limit content to 250 characters
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper) # Creates a tool that can search academic papers on Arxiv

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper) # Creates a tool for searching Wikipedia articles

search = DuckDuckGoSearchRun(name="Search") # inbuilt general web search tool using DuckDuckGo

load_dotenv()
os.environ['SSL_CERT_FILE'] = certifi.where()
st.title("🔎 LangChain - Chat with search")

## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Set default API key if none provided
if not api_key:
    api_key = "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"

# Initializes the chat history in session state if it doesn't exist
# Adds a welcome message from the assistant
# Displays all existing messages from the chat history
if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt := st.chat_input(placeholder="What is machine learning?"): # Creates a chat input field with a placeholder(temporary) suggestion
    st.session_state.messages.append({"role": "user", "content": prompt}) # When the user submits a message: Adds it to the chat history & Displays it in the chat interface
    st.chat_message("user").write(prompt)


    # Initializes the llama-4-scout-17b-16e-instruct model from Groq
    # Combines all three search tools into a list & creates an agent that:
    # - Uses the ZERO_SHOT_REACT_DESCRIPTION approach (makes decisions without prior examples)
    # - Handles parsing errors gracefully by returning raw LLM output if the format is invalid
    llm = ChatGroq(groq_api_key=api_key, model_name="meta-llama/llama-4-scout-17b-16e-instruct", streaming=True)
    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, # it gives output without relying on the chat history, that is on the basis of the current input
        handle_parsing_errors=True # handle_parsing_errors=True :- If the LLM outputs an invalid format, so instead of crashing, the agent will Catch the parsing error & will return the raw LLM output as a final answer. This acts as a fallback mechanism
    )

    # - Creates a chat message container for the assistant's response
    # - Initializes the StreamlitCallbackHandler to display the agent's thought process
    # - Runs the agent with the user's prompt
    # - Adds the response to the chat history
    # - Displays the final response
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True) # StreamlitCallbackHandler is showing the actions of an agent in streamlit app
        response = search_agent.run(prompt)
        st.session_state.messages.append({'role': 'assistant', "content": response})
        st.write(response)
