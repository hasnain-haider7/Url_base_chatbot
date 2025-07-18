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