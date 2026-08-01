```markdown
# LUCA

LUCA is an AI multi-agent workflow app built with LangGraph and Streamlit.  
It coordinates multiple specialized agents such as planner, researcher, coder, reviewer, and writer to process a user task step by step.

## Overview

LUCA is designed to:
- understand the user task
- gather useful research if needed
- generate code
- review the code
- write the final response
- automatically route between agents using a graph-based workflow

It also supports model fallback so that if one model hits a rate limit, another model can be used automatically.

## Features

- Multi-agent orchestration with LangGraph
- Planner-based routing logic
- Research, coding, review, and writing stages
- Automatic retry handling for coder loops
- Model fallback support
- Streamlit web UI
- Environment-based configuration with `.env`
- Deployment-ready structure

## State Graph

Add your stategraph here:

<!-- Replace this with your actual graph image later -->
![State Graph](![alt text](image.png))

You can also describe the flow here once the graph is ready:

- User input enters the planner
- Planner decides whether to route to:
  - researcher
  - coder
  - reviewer
  - writer
  - end
- The graph continues until the task is complete

## How It Works

### 1. Planner
The planner reads the current task and the available state, then decides which agent should run next.

### 2. Researcher
The researcher gathers useful facts, constraints, tools, or implementation details.

### 3. Coder
The coder generates the actual code or technical output needed for the task.

### 4. Reviewer
The reviewer checks the code or output and either approves it or asks for fixes.

### 5. Writer
The writer creates the final polished response for the user.

## Project Structure

```text
LUCA/
├── main.py
├── pyproject.toml
├── req.txt
├── README.md
├── .gitignore
├── .env
├── .env.example
├── uv.lock
└── src/
    ├── app.py
    ├── config.py
    ├── agents/
    │   ├── planner.py
    │   ├── researcher.py
    │   ├── coder.py
    │   ├── reviewer.py
    │   └── writer.py
    └── graph/
        ├── state.py
        └── workflow.py

```

## Installation

### 1. Clone the repository

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO.git](https://github.com/YOUR_USERNAME/YOUR_REPO.git)
cd LUCA

```

### 2. Create and activate a virtual environment

If you are using `uv`:

```bash
uv venv

```

Activate it on Windows:

```bash
.venv\Scripts\activate

```

### 3. Install dependencies

If you are using `uv`:

```bash
uv sync

```

If you are using pip:

```bash
pip install -r req.txt

```

## Environment Variables

Create a .env file in the project root with the following values:

```env
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
LUCA_GROQ_MODELS=openai/gpt-oss-20b,llama-3.1-8b-instant,llama-3.1-70b-versatile

```

### Variable descriptions

* `GROQ_API_KEY`: required for Groq model access
* `LANGCHAIN_API_KEY`: optional, used for LangSmith tracing
* `LUCA_GROQ_MODELS`: comma-separated list of fallback models

## Running Locally

Start the app with Streamlit:

```bash
streamlit run src/app.py

```

If needed, you can also run the Python entrypoint:

```bash
python main.py

```

## Deployment

This app can be deployed on:

* Streamlit Community Cloud
* Render
* Any platform that supports Python and Streamlit

### Deployment notes

Make sure your deployment environment includes:

* `GROQ_API_KEY`
* `LANGCHAIN_API_KEY` if tracing is enabled
* `LUCA_GROQ_MODELS`

Also make sure dependencies are installed from:

* pyproject.toml
* uv.lock
or
* req.txt

## Example Workflow

1. User submits a task
2. Planner evaluates the state
3. Planner routes to the correct agent
4. Researcher gathers supporting information if needed
5. Coder generates code
6. Reviewer checks the output
7. Planner decides whether to retry or finish
8. Writer returns the final answer

## Troubleshooting

### 429 rate limit error

If you see a rate limit error:

* check your model quota
* make sure `LUCA_GROQ_MODELS` includes fallback models
* reduce prompt size if needed

### App does not start

Check:

* .env exists
* API keys are valid
* dependencies are installed
* the correct Streamlit file is being used

### Graph loops too long

Check:

* planner routing logic
* retry counter handling
* reviewer output format
* recursion limit

## Future Improvements

* Add better graph visualization
* Add more agent roles
* Add persistent logging
* Add tests for routing behavior
* Add a more advanced fallback strategy
* Improve prompt templates for each agent

## License

Add your license here if needed.

## Author

Created with LUCA.

```

```