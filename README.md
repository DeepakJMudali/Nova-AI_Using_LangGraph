# Nova AI

Nova AI is a Streamlit-based conversational assistant powered by LangGraph, Groq, and Tavily Search. It provides a polished chat interface for asking questions, exploring ideas, and retrieving live web information during the conversation.

## Features

- Conversational AI chat interface built with Streamlit
- Persistent conversation memory using LangGraph checkpointers
- Web search tool integration via Tavily
- Groq-hosted LLM integration
- Dark, modern UI tailored for an AI workspace experience

## Tech Stack

- Python 3.13+
- Streamlit
- LangGraph
- LangChain
- Groq LLM
- Tavily Search
- python-dotenv

## Project Structure

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── src/
│   └── agentiai_project/
│       ├── __init__.py
│       └── app.py
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/DeepakJMudali/Nova-AI_Using_LangGraph.git
cd Nova-AI_Using_LangGraph
```

### 2. Create a virtual environment

Using venv:

```bash
python -m venv .venv
source .venv/bin/activate   # On macOS/Linux
.\.venv\Scripts\activate    # On Windows
```

### 3. Install dependencies

Using pip:

```bash
pip install -r requirements.txt
```

Or with uv:

```bash
uv sync
```

### 4. Set up environment variables

Create a `.env` file in the project root with your API keys:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

The app loads these values automatically using `python-dotenv`.

## Running the App

From the project root, start the Streamlit app:

```bash
streamlit run src/agentiai_project/app.py
```

Then open the local URL shown by Streamlit in your browser.

## Usage

- Type a question in the chat box
- The assistant can answer using the configured language model
- If needed, it can use Tavily to perform web search and incorporate the results into the answer
- Each chat session keeps its own thread state and memory

## Notes

- This project uses `LangGraph` for agent orchestration and state handling.
- The Groq model is configured in `src/agentiai_project/app.py` and can be adjusted if needed.
- The app currently exposes a single chat workflow with web-search capability.

## License

This project does not currently include a license file. If you plan to distribute or publish it, add an appropriate open-source license before release.
