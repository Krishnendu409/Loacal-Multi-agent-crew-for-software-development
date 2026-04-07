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
| 🧠 **Built-in skill packs** | Shared + role-specific skills are injected into prompts, configurable in `config.yaml` |
| 👥 **Full dev team** | CEO Planner → Market Researcher → PM → Architect → Frontend + Backend → QA → Code Reviewer → DevOps (optional) |
| 🔁 **Fix-pass loop** | Dev → QA/Reviewer → Dev remediation pass (bounded iterations) |
| 📄 **Saved outputs** | Every agent's response and a final compiled report saved to `output/` |
| ⚙️ **Configurable** | Swap models, enable/disable agents, change output paths via `config.yaml` |

---

## 🏗️ Architecture

```
problem statement (you)
        │
        ▼
┌─────────────────┐
│   CEO Planner   │  Creates strategy + execution roadmap
└────────┬────────┘
         │  context
         ▼
┌─────────────────┐
│Market Researcher│  Finds market gaps and differentiation
└────────┬────────┘
         │  context
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
│Frontend Developer│ Designs and builds UX/UI layer
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
back to Frontend/Backend for a remediation pass. A strategy approval gate can
pause the run after planning so the user can explicitly approve before build work.

---

## 🚀 Quick Start

### 1. Install Ollama

Download and install [Ollama](https://ollama.com/download) for your OS, then
pull a free model:

```bash
# Reasoning model
ollama pull qwen2.5:7b-instruct

# Coding model
ollama pull deepseek-coder

# Critic model
ollama pull phi3:mini
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
  model: qwen2.5:7b-instruct
  base_url: http://localhost:11434
  retries: 1
  timeout_seconds: 120
  allowed_models:
    - qwen2.5:7b-instruct
    - deepseek-coder:6.7b
    - phi3:mini
  options:
    temperature: 0.4
    num_predict: 2048
  routing:
    ceo_planner: qwen2.5:7b-instruct
    market_researcher: qwen2.5:7b-instruct
    product_manager: qwen2.5:7b-instruct
    architect: qwen2.5:7b-instruct
    frontend_developer: deepseek-coder:6.7b
    backend_developer: deepseek-coder:6.7b
    qa_engineer: phi3:mini
    code_reviewer: phi3:mini
    devops_engineer: deepseek-coder:6.7b
  fallbacks:
    ceo_planner: [phi3:mini]
    market_researcher: [phi3:mini]
    backend_developer: [qwen2.5:7b-instruct]
  role_options:
    backend_developer: {temperature: 0.2, num_predict: 2048}
    qa_engineer: {temperature: 0.1, num_predict: 1024}
    code_reviewer: {temperature: 0.1, num_predict: 1024}

agents:
  ceo_planner: true
  market_researcher: true
  product_manager: true
  architect: true
  frontend_developer: true
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
  require_strategy_approval: true

skills:
  include_default_role_skills: true
  enforce_handoff_sections: true
  strict_mode: false
  max_skills_per_agent: 12
  packs:
    ecc:
      enabled: true
      profile: starter
  shared:
    - structured communication
    - security-first thinking
    - verification mindset
    - documentation discipline
  per_role:
    frontend_developer: [ui consistency and accessibility]
    backend_developer: [dependency hygiene]
    qa_engineer: [risk-based test prioritization]
  include: []
  per_role_include: {}
  exclude: []
  per_role_exclude: {}
```

---

## 🧩 Imported ECC skill packs

The project now supports an external imported pack from
`affaan-m/everything-claude-code` via `skills.packs.ecc`.

- `profile: starter` → high-impact default set (recommended)
- `profile: advanced` → broader coverage set
- `max_skills_per_agent` enforces prompt budget by keeping highest-priority skills
- `exclude` / `per_role_exclude` disable noisy skills
- `include` / `per_role_include` force-enable custom additions
- `strict_mode: true` validates include/exclude references and fails on unknown skills

Inventory and mapping artifacts:
- `src/skills/data/ecc_inventory.json`
- `src/skills/data/ecc_traceability_matrix.json`

### Migration notes

- Existing keys (`include_default_role_skills`, `enforce_handoff_sections`,
  `shared`, `per_role`) remain supported and keep previous behavior.
- New pack skills are layered in as: built-in defaults → external pack → user overrides.
- If prompts become too long, reduce `max_skills_per_agent`, switch to
  `starter`, or add `exclude` entries.

---

## 📁 Project Structure

```
├── main.py                      # CLI entry point
├── config.yaml                  # User configuration
├── requirements.txt
├── src/
│   ├── agents/
│   │   ├── base_agent.py        # Agent base class
│   │   └── definitions.py       # CEO/Research/PM/Arch/FE/BE/QA/Reviewer role definitions
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

## 💡 Model Policy

This project is configured to use only:

- `qwen2.5:7b-instruct` (reasoning/planning)
- `deepseek-coder:6.7b` (implementation/frontend/backend/debugging)
- `phi3:mini` (critique/QA/review)

---

## 📄 License

MIT
