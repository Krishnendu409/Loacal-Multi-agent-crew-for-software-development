# 🤖 Local Multi-Agent Software Development Crew

A **free**, **fully local**, multi-agent system that simulates a complete software
development team on your laptop.  No cloud APIs, no subscription fees, no data
sent off-device.  Just a local [Ollama](https://ollama.com) model and a Python
runtime.

---

## ✨ Features

| Feature | Detail |
|---------|--------|
| 🆓 **Free** | Uses local Ollama models – zero cost, zero API keys |
| 💻 **Runs locally** | Works on mid-range laptops (4 GB RAM minimum with `phi3` or `llama3.2`) |
| 🗣️ **Agent communication** | Each agent reads compressed prior context + original requirements + QA/Reviewer checklists |
| 👥 **Full dev team** | PM → Architect → Developer → QA → Code Reviewer → DevOps (optional) |
| 🔁 **Fix-pass loop** | Dev → QA/Reviewer → Dev remediation pass (bounded iterations) |
| 📄 **Saved outputs** | Every agent's response and a final compiled report saved to `output/` |
| ⚙️ **Configurable** | Swap models, enable/disable agents, change output paths via `config.yaml` |

---

## 🏗️ Architecture

```
requirements (you)
        │
        ▼
┌─────────────────┐
│  Product Manager│  Analyses requirements → product spec
└────────┬────────┘
         │  context
         ▼
┌─────────────────┐
│   Architect     │  Designs system architecture
└────────┬────────┘
         │  context
         ▼
┌─────────────────┐
│Backend Developer│  Implements the code
└────────┬────────┘
         │  context
         ▼
┌─────────────────┐
│   QA Engineer   │  Writes test plan & automated tests
└────────┬────────┘
         │  context
         ▼
┌─────────────────┐
│  Code Reviewer  │  Reviews everything for quality & security
└────────┬────────┘
         │  (optional)
         ▼
┌─────────────────┐
│ DevOps Engineer │  Dockerfile, CI/CD, runbook
└─────────────────┘
```

Every agent receives **original requirements + compressed accumulated context**
from previous agents. QA/Reviewer can emit a must-address checklist that is fed
back to the Developer for a remediation pass.

---

## 🚀 Quick Start

### 1. Install Ollama

Download and install [Ollama](https://ollama.com/download) for your OS, then
pull a free model:

```bash
# Recommended: lightweight and fast on mid-range laptops (~4 GB RAM)
ollama pull mistral

# Ultra-lightweight option (~2 GB RAM)
ollama pull llama3.2

# Best code quality (~4 GB RAM)
ollama pull deepseek-coder
```

Start the Ollama daemon (if it is not already running):

```bash
ollama serve
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the crew

**Interactive mode** (will prompt you for project name and requirements):

```bash
python main.py run
```

**Inline mode**:

```bash
python main.py run \
  --project "Todo API" \
  --requirements "Build a REST API for a todo list application with CRUD operations"
```

**Use a different model for this run**:

```bash
python main.py run --model llama3.2 --project "Todo API"
```

---

## ⚙️ Configuration

Copy or edit `config.yaml` to customise the system:

```yaml
llm:
  model: mistral          # global fallback model (routing below is preferred)
  base_url: http://localhost:11434
  retries: 1
  timeout_seconds: 120
  options:
    temperature: 0.4
    num_predict: 2048
  routing:
    product_manager: qwen2.5:7b-instruct
    architect: qwen2.5:7b-instruct
    backend_developer: qwen2.5-coder:7b
    qa_engineer: phi3:mini
    code_reviewer: phi3:mini
    devops_engineer: llama3.2:3b
  fallbacks:
    backend_developer: [deepseek-coder:6.7b, llama3.2:3b]
  role_options:
    backend_developer: {temperature: 0.2, num_predict: 2048}
    qa_engineer: {temperature: 0.1, num_predict: 1024}
    code_reviewer: {temperature: 0.1, num_predict: 1024}

agents:
  product_manager: true
  architect: true
  backend_developer: true
  qa_engineer: true
  code_reviewer: true
  devops_engineer: false  # set to true to add deployment configs

output:
  directory: output
  save_individual_responses: true
  save_final_report: true

crew:
  max_fix_iterations: 1
  stop_on_no_major_issues: true
```

---

## 📁 Project Structure

```
├── main.py                      # CLI entry point
├── config.yaml                  # User configuration
├── requirements.txt
├── src/
│   ├── agents/
│   │   ├── base_agent.py        # Agent base class
│   │   └── definitions.py       # All role definitions & build_agents()
│   ├── tasks/
│   │   └── software_dev_tasks.py  # Task templates for each role
│   ├── crew/
│   │   └── dev_crew.py          # Pipeline orchestrator
│   ├── config/
│   │   └── settings.py          # Config loader
│   └── utils/
│       ├── display.py           # Rich terminal UI
│       └── ollama_client.py     # Ollama API wrapper
├── tests/
│   ├── test_agents.py
│   ├── test_config.py
│   ├── test_crew.py
│   └── test_tasks.py
└── output/                      # Generated artefacts (git-ignored)
```

---

## 🛠️ Other CLI Commands

```bash
# List locally available Ollama models
python main.py models

# Show resolved configuration (defaults merged with config.yaml)
python main.py config

# Use a custom config file
python main.py run --config path/to/my_config.yaml
```

---

## 🧪 Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## 💡 Recommended Models by Hardware

| RAM available | Model | Pull command |
|---------------|-------|--------------|
| 2 GB | `phi3` | `ollama pull phi3` |
| 2 GB | `llama3.2` | `ollama pull llama3.2` |
| 4 GB | `mistral` (default) | `ollama pull mistral` |
| 4 GB | `deepseek-coder` | `ollama pull deepseek-coder` |
| 8 GB | `llama3` | `ollama pull llama3` |
| 8 GB | `codestral` | `ollama pull codestral` |

---

## 📄 License

MIT
