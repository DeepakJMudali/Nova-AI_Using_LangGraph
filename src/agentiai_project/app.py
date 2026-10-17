import uuid
from typing import Annotated

import streamlit as st
from dotenv import load_dotenv
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.messages import trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Nova AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #0b0d17;
        --panel: #14182a;
        --border: #282d45;
        --text: #f0f2ff;
        --muted: #a5aec8;
        --accent: #a78bfa;
    }

    .stApp {
        background:
            radial-gradient(
                ellipse at 50% 0%,
                rgba(112, 76, 220, .15),
                transparent 48%
            ),
            var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Inter', sans-serif !important;
    }

    .block-container {
        max-width: 950px;
        padding-top: 2rem;
        padding-bottom: 6rem;
    }

    #MainMenu,
    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0e1120 !important;
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.8rem;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: #dfe3f7 !important;
    }

    /* Typography */
    h1, h2, h3, p {
        font-family: 'Inter', sans-serif !important;
    }

    h1 {
        font-weight: 800 !important;
        letter-spacing: -1.3px;
    }

    /* Chat message cards */
    [data-testid="stChatMessage"] {
        background: #14182a !important;
        border: 1px solid #282d45 !important;
        border-radius: 18px !important;
        padding: 1rem 1.15rem !important;
        margin-bottom: .9rem !important;
        color: #f0f2ff !important;
        opacity: 1 !important;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {
        color: #f0f2ff !important;
        opacity: 1 !important;
    }

    /* Chat input container */
    [data-testid="stChatInput"] {
        background: #15182a !important;
        border: 1px solid #34304f !important;
        border-radius: 18px !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] > div {
        background: transparent !important;
        border-radius: 18px !important;
    }

    /* Typed text: mobile and desktop */
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea:focus {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        caret-color: #ffffff !important;
        background: transparent !important;
        opacity: 1 !important;
        font-size: 16px !important;
    }

    /* Placeholder */
    [data-testid="stChatInput"] textarea::placeholder {
        color: #a5aec8 !important;
        -webkit-text-fill-color: #a5aec8 !important;
        opacity: 1 !important;
    }

    /* Input focus effect */
    [data-testid="stChatInput"]:focus-within {
        border-color: #a78bfa !important;
        box-shadow: 0 0 0 2px rgba(167, 139, 250, .15) !important;
    }

    /* Mobile browser autofill */
    [data-testid="stChatInput"] textarea:-webkit-autofill,
    [data-testid="stChatInput"] textarea:-webkit-autofill:hover,
    [data-testid="stChatInput"] textarea:-webkit-autofill:focus {
        -webkit-text-fill-color: #ffffff !important;
        caret-color: #ffffff !important;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        min-height: 43px;
        background: #171b2e !important;
        color: #f0f2ff !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        transition: all .2s ease;
    }

    .stButton > button:hover {
        background: #25203e !important;
        border-color: var(--accent) !important;
        color: #ffffff !important;
    }

    /* Welcome hero */
    .hero {
        padding: 2rem;
        margin: 1.1rem 0 1.8rem;
        border-radius: 24px;
        border: 1px solid #302b50;
        background: linear-gradient(
            135deg,
            rgba(125, 91, 230, .17),
            rgba(50, 91, 165, .08)
        );
    }

    .hero-tag {
        color: #c4b5fd;
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: 2px;
    }

    .hero-title {
        color: #f5f3ff !important;
        font-size: 2.4rem;
        line-height: 1.2;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin: .8rem 0;
    }

    .hero-subtitle {
        color: #c0c7df !important;
        font-size: .98rem;
        line-height: 1.8;
    }

    .section-caption {
        color: #b9c1db !important;
        font-size: .88rem;
        margin-bottom: 1rem;
    }

    /* Sidebar branding */
    .sidebar-brand {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -1px;
        color: #f5f3ff;
        margin-bottom: .25rem;
    }

    .sidebar-subtitle {
        color: #b0b8d2;
        font-size: .82rem;
        margin-bottom: 1.5rem;
    }

    /* Status indicator */
    .status {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 20px;
        border: 1px solid rgba(52, 211, 153, .24);
        background: rgba(52, 211, 153, .08);
        color: #6ee7b7 !important;
        font-size: .76rem;
    }

    hr {
        border-color: var(--border) !important;
    }

    /* Mobile layout */
    @media (max-width: 640px) {
        .block-container {
            padding: 1rem 1rem 5rem;
        }

        .hero {
            padding: 1.3rem;
        }

        .hero-title {
            font-size: 1.8rem;
        }

        .hero-subtitle {
            font-size: .9rem;
        }

        [data-testid="stChatMessage"] {
            padding: .85rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# --------------------------------------------------
# LangGraph state
# --------------------------------------------------
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# --------------------------------------------------
# Create graph once
# --------------------------------------------------
@st.cache_resource
def create_graph():
    llm = ChatGroq(model="openai/gpt-oss-20b")

    search_tool = TavilySearch(max_results=2)
    tools = [search_tool]

    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: State):
        messages = trim_messages(
        state["messages"],
        max_tokens=4000,
        strategy="last",
        token_counter=llm,
        include_system=True,
        allow_partial=False,
        start_on="human",
       )
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(State)

    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", tools_condition)
    builder.add_edge("tools", "chatbot")

    return builder.compile(checkpointer=MemorySaver())


graph = create_graph()


# --------------------------------------------------
# Thread ID: preserved in URL across refreshes
# --------------------------------------------------
thread_id = st.query_params.get("thread_id")

if not thread_id:
    thread_id = str(uuid.uuid4())
    st.query_params["thread_id"] = thread_id

config = {
    "configurable": {
        "thread_id": thread_id,
    }
}


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">✦ Nova AI</div>
        <div class="sidebar-subtitle">
            Your intelligent AI workspace
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<span class="status">● Assistant ready</span>',
                unsafe_allow_html=True)

    st.divider()

    st.markdown("### Conversation")

    if st.button("＋  Start a new chat", use_container_width=True):
        st.query_params["thread_id"] = str(uuid.uuid4())
        st.rerun()

    st.caption("Current conversation")
    st.code(thread_id[:12] + "...", language=None)

    st.divider()

    st.markdown("### Capabilities")
    st.markdown(
        """
        - 💬 Conversational AI
        - 🔎 Web search with Tavily
        - 🧠 LangGraph conversation memory
        """
    )

    st.divider()
    st.caption("Powered by Groq · LangGraph · Tavily")


# --------------------------------------------------
# Main header
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-tag">✦ YOUR AI WORKSPACE</div>
        <div class="hero-title">Meet Nova AI.</div>
        <div class="hero-subtitle">
            Ask questions, explore ideas, search the web,
            and solve problems — all in one conversation.
        </div>
        <br>
        <span class="status">● Ready to help</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Read existing conversation from graph state
# --------------------------------------------------
snapshot = graph.get_state(config)
saved_messages = snapshot.values.get("messages", [])

# Show welcome prompts only for a new conversation.
if not saved_messages:
    st.markdown("### What can we explore today?")
    st.markdown(
        '<div class="section-caption">'
        'Choose a starting point or type your own question below.'
        '</div>',
        unsafe_allow_html=True,
    )

    


# --------------------------------------------------
# Render saved messages
# --------------------------------------------------
for message in saved_messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):
        # Hide empty AI messages that only request tool calls.
        if message.content:
            with st.chat_message("assistant"):
                st.markdown(message.content)

    elif isinstance(message, ToolMessage):
        # Tool output remains in graph history for the LLM,
        # but is not shown as a separate chat bubble.
        pass


# --------------------------------------------------
# Chat input
# --------------------------------------------------
suggested_prompt = st.session_state.pop("suggested_prompt", None)

user_input = st.chat_input("Message Nova AI...")

if suggested_prompt:
    user_input = suggested_prompt

if user_input:
    # Show the new user message immediately.
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Nova is thinking..."):
            try:
                result = graph.invoke(
                    {
                        "messages": [
                            HumanMessage(content=user_input)
                        ]
                    },
                    config=config,
                )

                # Find the last non-empty AI response.
                answer = None

                for message in reversed(result["messages"]):
                    if isinstance(message, AIMessage) and message.content:
                        answer = message.content
                        break

                if answer is not None:
                    st.markdown(answer)
                else:
                    st.warning("The assistant returned no text response.")

            except Exception as e:
                st.error("Something went wrong while generating the response.")
                st.exception(e)
