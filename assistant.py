# uobs_assistant.py

# ========== ENVIRONMENT SETUP ==========
import os
from dotenv import load_dotenv
load_dotenv()

# API Keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Not used

# ========== LLM & TOOLS ==========
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

SPECIFIC_SEARCH_URLS = [
    "uobs.edu.pk",
    "uobs.edu.pk/admissions",
    "uobs.edu.pk/affiliated-colleges",
    "uobs.edu.pk/admissions?view=article&id=142:academic-calander-and-semester-plan&catid=40",
    "uobs.edu.pk/admissions?view=article&id=253:fee-structure&catid=40",
    "https://uobs.edu.pk/admissions?view=article&id=141:admission-policy&catid=40",
    "https://uobs.edu.pk/admissions?view=article&id=294:2nd-merit-list-admission-spring-2025&catid=40",
    "https://uobs.edu.pk/faculties/faculty-of-natural-sciences/computer-science?view=article&id=90:members&catid=17",
    "https://uobs.edu.pk/faculties/faculty-of-natural-sciences/computer-science?view=article&id=91:offered-courses&catid=17",
    "https://uobs.edu.pk/faculties/faculty-of-natural-sciences/computer-science?view=article&id=93:uobs-computing-society&catid=17",
    "https://uobs.edu.pk/faculties/faculty-of-natural-sciences/computer-science?view=article&id=92:scheme-of-study&catid=17"
]

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt):
    llm = ChatGroq(model=llm_id)
    tools = []

    if allow_search:
        tools.append(TavilySearchResults(max_results=2, include_domains=SPECIFIC_SEARCH_URLS))

    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )
    state = {"messages": query}
    response = agent.invoke(state)
    messages = response.get("messages")
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1]

# ========== FASTAPI BACKEND ==========
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

class RequestState(BaseModel):
    model_name: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

app = FastAPI(title="LangGraph AI Agent")
ALLOWED_MODEL_NAMES = ["llama3-70b-8192", "mixtral-8x7b-32768", "llama-3.3-70b-versatile"]

@app.post("/chat")
def chat_endpoint(request: RequestState):
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": "Invalid model name. Kindly select a valid AI model"}
    response = get_response_from_ai_agent(request.model_name, request.messages, request.allow_search, request.system_prompt)
    return response

# ========== STREAMLIT FRONTEND ==========
import streamlit as st
import requests
import json
import threading

def run_streamlit():
    st.set_page_config(page_title="UOBS Assistant", layout="centered")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Agent Settings")
        system_prompt = st.text_area("Agent Act :", height=100, placeholder="Type your system prompt here...")
        MODEL_NAMES_GROQ = ["llama3-70b-8192", "mixtral-8x7b-32768", "llama-3.3-70b-versatile"]
        selected_model = st.selectbox("Choose Model:", MODEL_NAMES_GROQ)
        allow_web_search = st.checkbox("Web Search")
        st.write("---")
        st.caption("Configure these setting to get response.")

    st.title("Virtual Assistant")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Type a message...")
    API_URL = "http://127.0.0.1:9999/chat"

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        payload = {
            "model_name": selected_model,
            "system_prompt": system_prompt,
            "messages": [user_query],
            "allow_search": allow_web_search
        }

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                agent_response = "Error: Something went wrong with the agent's response."
                try:
                    response = requests.post(API_URL, json=payload, timeout=60)
                    if response.status_code == 200:
                        try:
                            response_data = response.json()
                            if isinstance(response_data, dict):
                                if "error" in response_data:
                                    agent_response = f"**Error:** {response_data['error']}"
                                    st.error(agent_response)
                                else:
                                    agent_response = response_data.get("response", str(response_data))
                                    st.markdown(agent_response)
                            elif isinstance(response_data, str):
                                agent_response = response_data
                                st.markdown(agent_response)
                            else:
                                agent_response = f"**Unexpected response format:** {type(response_data).__name__}"
                                st.error(agent_response)
                        except json.JSONDecodeError:
                            agent_response = response.text
                            st.markdown(f"**Agent Response (Plain Text):** {agent_response}")
                            st.warning("The backend did not return a valid JSON response.")
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

            st.session_state.messages.append({"role": "assistant", "content": agent_response})


# ========== MAIN LAUNCH ==========
if __name__ == "__main__":
    import uvicorn

    # Run FastAPI in a separate thread
    def run_fastapi():
        uvicorn.run(app, host="127.0.0.1", port=9999, log_level="info")

    thread = threading.Thread(target=run_fastapi, daemon=True)
    thread.start()

    # Run Streamlit app (note: for proper launching, Streamlit should ideally be launched via CLI)
    run_streamlit()
