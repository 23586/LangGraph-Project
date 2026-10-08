# LangGraph Project

[One or two sentences: what this project does, e.g. "An AI agent built with LangGraph that ..."]

## Architecture

![Architecture](architecture.png)

![Flow of LangGraph](Flow%20of%20LangGraph.png)

An interactive version is available in `architecture.html`.

## Project Structure

| File | Description |
|------|-------------|
| `agent.py` | Defines the LangGraph agent, its nodes, and the graph flow |
| `app.py` | Main entry point / user interface for running the app |
| `tools.py` | Tools the agent can call |
| `db.py` | Database setup and helpers (state is stored in a local SQLite DB) |
| `architecture.html` / `architecture.png` | Architecture diagram |
| `Flow of LangGraph.png` | Diagram of the graph flow |

## Getting Started

### Prerequisites
- Python 3.10+
- Qwen3:1.7b Local Model

### Running
```bash
python app.py
```
