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
| 👥 **Full dev team** | 18-role local team spanning strategy, market research, design, engineering, review, reliability, docs, and release (see architecture flow below) |
| 🔁 **Autonomous fix loop** | Generate code → write files → execute tests → critique → remediate (bounded iterations) |
| 📄 **Saved outputs** | Versioned structured artifacts, generated project files, and execution memory |
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
┌───────────────────────────────┐
│Customer Support/Feedback Anal.│  Converts pain points into requirements
└────────┬──────────────────────┘
         │  context
         ▼
┌─────────────────┐
│  Product Manager│  Finalizes strategy + product spec
└────────┬────────┘
         │  context
         ▼
┌───────────────────────────────┐
│Compliance & Privacy Specialist│  Defines mandatory controls
└────────┬──────────────────────┘
         │
         ▼
     [USER APPROVAL GATE]
         │
         ▼
┌─────────────────┐
│   Architect     │  Designs system architecture
└────────┬────────┘
         │  context
         ▼
┌─────────────────┐
│  UI/UX Designer │  Defines journeys, interactions, and accessibility
└────────┬────────┘
         │  context
         ▼
┌─────────────────┐
│Database Engineer│  Defines schema/index/migration strategy
└────────┬────────┘
         │  context
         ▼
┌───────────────────────┐
│API Integration Engineer│  Defines resilient API contracts
└────────┬──────────────┘
         │  context
         ▼
┌─────────────────┐
│Frontend Developer│ Designs and builds UX/UI layer
└────────┬────────┘
         │  context
         ▼
┌─────────────────┐
│Backend Developer│  Implements core logic
└────────┬────────┘
         │  context
         ▼
┌────────────────────┐
│Data/Analytics Eng. │  Defines KPI/event instrumentation
└────────┬───────────┘
         │  context
         ▼
┌─────────────────┐
│Performance Eng. │  Finds bottlenecks and tuning priorities
└────────┬────────┘
         │  context
         ▼
┌─────────────────┐
│Security Engineer│  Threat-models and reviews vulnerabilities
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
         │  context
         ▼
┌─────────────────┐
│Technical Writer │  Creates docs + runbooks
└────────┬────────┘
         │  context
         ▼
┌──────────────────────┐
│SRE / Reliability Eng.│  Defines SLOs and incident readiness
└────────┬─────────────┘
         │  context
         ▼
┌─────────────────┐
│ Release Manager │  Final go/no-go release plan
└────────┬────────┘
         │  (optional)
         ▼
┌─────────────────┐
│ DevOps Engineer │  Dockerfile, CI/CD, runbook
└─────────────────┘
```

Every agent receives **original requirements + compressed accumulated context**
from previous agents. Performance/Security/QA/Reviewer can emit must-address
checklists that are fed back to implementation roles for a remediation pass.
The strategy phase is explicitly finalized with user approval before architecture
and build work starts.

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

By default, strategy approval requires explicit confirmation (`Continue` defaults to `No`).
Use `--auto-approve-strategy` to proceed non-interactively.

**Inline mode**:

```bash
python main.py run \
  --project "Todo API" \
  --requirements "Build a REST API for a todo list application with CRUD operations"
```

**Resume from a specific role using previous outputs**:

```bash
python main.py run \
  --project "Todo API" \
  --requirements "Build a REST API for a todo list application with CRUD operations" \
  --resume-run-dir output/todo_api_20260408_120000 \
  --start-from-role "Security Engineer"
```

**Use a different model for this run**:

```bash
python main.py run --model deepseek-coder:6.7b --project "Todo API"
```

---

## ⚙️ Configuration

Copy or edit `config.yaml` to customise the system:

Environment overrides are also supported:
- `OLLAMA_BASE_URL` (legacy alias: `OLLAMA_URL`)
- `OLLAMA_MODEL`
- `MODEL_REASONING` (role-group override for planning/reasoning roles)
- `MODEL_CODING` (role-group override for implementation roles)
- `MODEL_CRITIC` (role-group override for review/specialist roles)
- `OLLAMA_RETRIES`
- `OLLAMA_TIMEOUT`
- `OLLAMA_TEMPERATURE`
- `OLLAMA_NUM_PREDICT`
- `CREW_MAX_FIX_ITERATIONS`
- `CREW_STOP_ON_NO_MAJOR_ISSUES`
- `CREW_REQUIRE_STRATEGY_APPROVAL`
- `OUTPUT_DIR`

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
    customer_support_feedback_analyst: qwen2.5:7b-instruct
    product_manager: qwen2.5:7b-instruct
    compliance_privacy_specialist: phi3:mini
    architect: qwen2.5:7b-instruct
    ui_ux_designer: qwen2.5:7b-instruct
    database_engineer: deepseek-coder:6.7b
    api_integration_engineer: deepseek-coder:6.7b
    frontend_developer: deepseek-coder:6.7b
    backend_developer: deepseek-coder:6.7b
    data_analytics_engineer: deepseek-coder:6.7b
    performance_engineer: phi3:mini
    security_engineer: phi3:mini
    qa_engineer: phi3:mini
    code_reviewer: phi3:mini
    technical_writer: qwen2.5:7b-instruct
    sre_reliability_engineer: deepseek-coder:6.7b
    release_manager: qwen2.5:7b-instruct
    devops_engineer: deepseek-coder:6.7b
  fallbacks:
    ceo_planner: [phi3:mini]
    market_researcher: [phi3:mini]
    backend_developer: [qwen2.5:7b-instruct]
  role_options:
    backend_developer: {temperature: 0.2, num_predict: 2048}
    qa_engineer: {temperature: 0.1, num_predict: 1024}
    code_reviewer: {temperature: 0.1, num_predict: 1024}
  role_retries: {}

agents:
  ceo_planner: true
  market_researcher: true
  customer_support_feedback_analyst: true
  product_manager: true
  compliance_privacy_specialist: true
  architect: true
  ui_ux_designer: true
  database_engineer: true
  api_integration_engineer: true
  frontend_developer: true
  backend_developer: true
  data_analytics_engineer: true
  performance_engineer: true
  security_engineer: true
  qa_engineer: true
  code_reviewer: true
  technical_writer: true
  sre_reliability_engineer: true
  release_manager: true
  devops_engineer: false  # set to true to add deployment configs

output:
  directory: output
  save_individual_responses: true
  save_final_report: true

crew:
  max_fix_iterations: 1
  stop_on_no_major_issues: true
  require_strategy_approval: true
  blocking_severities: [critical, major]
  research_mode: false
  research_urls: []
  research_timeout_seconds: 10
  research_max_chars_per_source: 2000

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
    customer_support_feedback_analyst: [support signal prioritization]
    compliance_privacy_specialist: [privacy impact thinking]
    ui_ux_designer: [interaction design and prototype clarity]
    database_engineer: [schema evolution safety]
    api_integration_engineer: [resilient integration contracts]
    frontend_developer: [ui consistency and accessibility]
    backend_developer: [dependency hygiene]
    data_analytics_engineer: [measurement reliability]
    performance_engineer: [evidence-driven optimization]
    security_engineer: [threat-led validation]
    qa_engineer: [risk-based test prioritization]
    technical_writer: [high-signal operational documentation]
    sre_reliability_engineer: [slo-first reliability planning]
    release_manager: [go-no-go discipline]
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
│   │   └── definitions.py       # Full role definitions across strategy, build, review, and release
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

Each run also writes `RUN_MANIFEST.json` in the run output folder with per-role
timing, model, retry, status, and structured section metadata.

---

## ✅ Quality Checks

```bash
pip install ruff mypy pip-audit
python -m ruff check .
python -m ruff format --check .
python -m mypy src main.py
pip-audit
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

---

## 🤝 Contributing and Project Policy

- Contribution guide: `CONTRIBUTING.md`
- Project roadmap: `ROADMAP.md`
- PR template: `.github/pull_request_template.md`
- Issue templates: `.github/ISSUE_TEMPLATE/`
