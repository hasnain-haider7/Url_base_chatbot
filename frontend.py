import streamlit as st
import requests
import json # Import json module to potentially handle JSONDecodeError if backend sends invalid JSON

st.set_page_config(page_title="Virtual Assistant", layout="centered")

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