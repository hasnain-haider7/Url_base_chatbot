# ========== FILE 1 START ==========

# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

#Step1: Setup API Keys for Groq, OpenAI and Tavily
import os

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")
# OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY") # Removed as OpenAI is no longer used

# Step2: Setup LLM & Tools
from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI # Removed as OpenAI is no longer used
from langchain_community.tools.tavily_search import TavilySearchResults # Or from langchain_tavily import TavilySearch for latest features

# Define the specific URLs for web search
SPECIFIC_SEARCH_URLS = [
    "uobs.edu.pk", # Base domain for all sub-paths
    "uobs.edu.pk/admissions",
    "uobs.edu.pk/affiliated-colleges",
    "uobs.edu.pk/admissions?view=article&id=142:academic-calander-and-semester-plan&catid=40",
    "uobs.edu.pk/admissions?view=article&id=253:fee-structure&catid=40"
    "https://uobs.edu.pk/admissions?view=article&id=141:admission-policy&catid=40"
    "https://uobs.edu.pk/admissions?view=article&id=294:2nd-merit-list-admission-spring-2025&catid=40"
    "https://uobs.edu.pk/admissions?view=article&id=142:academic-calander-and-semester-plan&catid=40"
    "https://uobs.edu.pk/faculties/faculty-of-natural-sciences/computer-science?view=article&id=90:members&catid=17"
    "https://uobs.edu.pk/faculties/faculty-of-natural-sciences/computer-science?view=article&id=91:offered-courses&catid=17"
    "https://uobs.edu.pk/faculties/faculty-of-natural-sciences/computer-science?view=article&id=93:uobs-computing-society&catid=17"
    "https://uobs.edu.pk/faculties/faculty-of-natural-sciences/computer-science?view=article&id=92:scheme-of-study&catid=17"
]

groq_llm=ChatGroq(model="llama-3.3-70b-versatile") # Keep Groq LLM instance for direct use if needed

# Step3: Setup AI Agent with Search tool functionality
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

system_prompt="Act as an AI chatbot who is smart and friendly"

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt): # Removed 'provider' parameter
    # Always use Groq as the provider
    llm = ChatGroq(model=llm_id)

    tools=[]
    if allow_search:
        # Initialize TavilySearchResults with include_domains
        tools.append(TavilySearchResults(max_results=2, include_domains=SPECIFIC_SEARCH_URLS))

    agent=create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )
    state={"messages": query}
    response=agent.invoke(state)
    messages=response.get("messages")
    ai_messages=[message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1]

# ========== FILE 1 END ==========


# ========== FILE 2 START ==========

# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

#Step1: Setup Pydantic Model (Schema Validation)
from pydantic import BaseModel
from typing import List


class RequestState(BaseModel):
    model_name: str
    # model_provider: str # Removed as provider is always Groq
    system_prompt: str
    messages: List[str]
    allow_search: bool


#Step2: Setup AI Agent from FrontEnd Request
from fastapi import FastAPI
# from ai_agent import get_response_from_ai_agent  # Already defined above

# Only Groq models allowed
ALLOWED_MODEL_NAMES=["llama3-70b-8192", "mixtral-8x7b-32768", "llama-3.3-70b-versatile"]

app=FastAPI(title="LangGraph AI Agent")

@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    API Endpoint to interact with the Chatbot using LangGraph and search tools.
    It dynamically selects the model specified in the request
    """
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": "Invalid model name. Kindly select a valid AI model"}

    llm_id = request.model_name
    query = request.messages
    allow_search = request.allow_search
    system_prompt = request.system_prompt
    # provider = "Groq" # Provider is implicitly Groq now

    # Create AI Agent and get response from it!
    response=get_response_from_ai_agent(llm_id, query, allow_search, system_prompt) # Removed 'provider' argument
    return response

# ========== FILE 2 END ==========


# ========== FILE 3 START ==========

import streamlit as st
import requests
import json # Import json module to potentially handle JSONDecodeError if backend sends invalid JSON

st.set_page_config(page_title="UOBS Assistant", layout="centered")

# Initialize chat history in session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar for settings (mimicking the "three dots" menu) ---
with st.sidebar:
    st.header("Agent Settings")
    system_prompt = st.text_area("Agent Act :", height=100, placeholder="Type your system prompt here...")

    # Only Groq models are available
    MODEL_NAMES_GROQ = ["llama3-70b-8192", "mixtral-8x7b-32768", "llama-3.3-70b-versatile"]

    # Remove provider selection and directly set to Groq
    provider = "Groq"
    selected_model = st.selectbox("Choose Model:", MODEL_NAMES_GROQ)

    allow_web_search = st.checkbox("Web Search")
    st.write("---")
    st.caption("Configure these setting to get response.")

# --- Main Chat Interface ---
st.title("Virtual Assistant")

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input for the user
user_query = st.chat_input("Type a message...")

API_URL = "http://127.0.0.1:9999/chat"

if user_query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Prepare payload for the backend
    payload = {
        "model_name": selected_model,
        # "model_provider": provider, # Removed from payload
        "system_prompt": system_prompt,
        "messages": [user_query], # Sending only the latest message for simplicity, backend should handle full history
        "allow_search": allow_web_search
    }

    # Display a "typing" indicator or similar while waiting for response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            agent_response = "Error: Something went wrong with the agent's response." # Default error message
            try:
                response = requests.post(API_URL, json=payload, timeout=60) # Added timeout for robustness
                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        if isinstance(response_data, dict):
                            if "error" in response_data:
                                agent_response = f"**Error:** {response_data['error']}"
                                st.error(agent_response)
                            else:
                                # Adjust "response" key if different, otherwise use string representation
                                agent_response = response_data.get("response", str(response_data))
                                st.markdown(agent_response)
                        elif isinstance(response_data, str):
                            # If response.json() directly returned a string
                            agent_response = response_data
                            st.markdown(agent_response)
                        else:
                            agent_response = f"**Unexpected response format:** {type(response_data).__name__}"
                            st.error(agent_response)

                    except json.JSONDecodeError:
                        # If the response is not valid JSON, treat it as plain text
                        agent_response = response.text
                        st.markdown(f"**Agent Response (Plain Text):** {agent_response}") # Indicate it's plain text for clarity
                        st.warning("The backend did not return a valid JSON response. Displaying as plain text.")
                    except Exception as e:
                        agent_response = f"**An error occurred while processing backend response:** {e}"
                        st.error(agent_response)
                else:
                    agent_response = f"**Error from backend:** Status Code {response.status_code} - {response.text}"
                    st.error(agent_response)
            except requests.exceptions.ConnectionError:
                agent_response = "**Error:** Could not connect to the backend server. Make sure it's running at " + API_URL
                st.error(agent_response)
            except requests.exceptions.Timeout:
                agent_response = "**Error:** The request timed out. The agent might be taking too long to respond."
                st.error(agent_response)
            except Exception as e:
                agent_response = f"**An unexpected error occurred:** {e}"
                st.error(agent_response)

        # Corrected line:
        st.session_state.messages.append({"role": "assistant", "content": agent_response})

# ========== FILE 3 END ==========


# ========== FILE 2 Step3: Run app ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)
