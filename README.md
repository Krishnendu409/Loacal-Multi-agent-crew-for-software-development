# 🤖 Local Multi-Agent Software Development Crew

> Free • Local • Runs on mid-range laptops (8 GB RAM, no GPU required)

A fully local, autonomous multi-agent system that builds complete software projects from a plain-English description — no cloud APIs, no subscriptions, no data leaving your machine.

---

## Quick Start

### 1. Install Ollama

Download and install [Ollama](https://ollama.com/download) for your OS, then pull the model:

```bash
ollama pull phi3:mini
```

> **Mid-range laptop?** `phi3:mini` is ~2.2 GB and runs on 8 GB RAM.  
> **16 GB+ RAM?** Pull `qwen2.5:7b-instruct` and `deepseek-coder:6.7b` for better code quality.

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
# Interactive mode (prompts for project name and requirements)
python main.py run

# Inline mode
python main.py run --project "Todo API" --requirements "Build a REST API for managing todos with SQLite"

# Skip strategy approval gate
python main.py run --project "Todo API" --requirements "..." --auto-approve-strategy

# List available Ollama models
python main.py models

# Show current resolved configuration
python main.py config
```

---

## Hardware Configurations

### 8 GB RAM (Mid-range Laptop) — Default Config

`config.yaml` is pre-tuned for this:
- Model: `phi3:mini` for all roles
- `enable_architect_quorum: false`
- `enable_system_runner: false`  
- `enable_vector_memory: false`
- `max_fix_iterations: 0`

### 16 GB RAM — Better Code Quality

Edit `config.yaml`:

```yaml
llm:
  model: qwen2.5:7b-instruct
  routing:
    # Planning roles
    ceo_planner: qwen2.5:7b-instruct
    market_researcher: qwen2.5:7b-instruct
    product_manager: qwen2.5:7b-instruct
    architect: qwen2.5:7b-instruct
    # Coding roles
    backend_developer: deepseek-coder:6.7b
    frontend_developer: deepseek-coder:6.7b
    database_engineer: deepseek-coder:6.7b
    # Critic roles stay on phi3:mini
    qa_engineer: phi3:mini
    code_reviewer: phi3:mini
  allowed_models:
    - qwen2.5:7b-instruct
    - deepseek-coder:6.7b
    - phi3:mini

crew:
  max_fix_iterations: 1
```

Pull the additional models:
```bash
ollama pull qwen2.5:7b-instruct
ollama pull deepseek-coder:6.7b
```

---

## Environment Variable Overrides

| Variable | Effect |
|---|---|
| `OLLAMA_MODEL` | Override the global model |
| `OLLAMA_BASE_URL` | Ollama server URL (default: `http://localhost:11434`) |
| `OLLAMA_TIMEOUT` | Inference timeout in seconds (default: 600) |
| `OLLAMA_TEMPERATURE` | Override temperature |
| `OLLAMA_NUM_PREDICT` | Override max tokens |
| `MODEL_REASONING` | Override all planning/strategy roles |
| `MODEL_CODING` | Override all implementation roles |
| `MODEL_CRITIC` | Override all reviewer roles |
| `CREW_MAX_FIX_ITERATIONS` | Override fix iteration count |
| `CREW_REQUIRE_STRATEGY_APPROVAL` | `false` to skip the strategy gate |
| `OUTPUT_DIR` | Where to save generated files (default: `output/`) |

---

## Output Structure

```
output/
└── my_project_20240410_143022/
    ├── FINAL_REPORT.md          # Full crew output in one document
    ├── RUN_MANIFEST.json        # Timing, model, status per agent
    ├── generated_project/       # Actual code files written by agents
    │   ├── src/
    │   ├── tests/
    │   └── ...
    ├── ceo_planner.md
    ├── product_manager.md
    ├── software_architect.md
    └── ... (one .md per agent)
```

---

## Troubleshooting

**"Could not reach the Ollama daemon"**  
Start Ollama: `ollama serve`

**"Model not found locally"**  
Pull it: `ollama pull phi3:mini`

**"Inference timed out"**  
Increase timeout: `OLLAMA_TIMEOUT=1200 python main.py run ...`  
Or edit `llm.timeout_seconds` in `config.yaml`.

**Model produces garbled JSON**  
Lower the token budget: set `num_predict: 512` in `llm.options` in `config.yaml`.  
This forces shorter, more focused outputs from small models.

**Out of memory**  
Ensure only one model is loaded at a time. Keep `enable_architect_quorum: false`.  
Quit other applications before running.

---

## Resuming a Run

If a run fails partway through, resume from where it stopped:

```bash
python main.py run \
  --project "My App" \
  --requirements "..." \
  --resume-run-dir output/my_app_20240410_143022 \
  --start-from-role "Backend Developer"
```

---

## License

MIT — see [LICENSE](LICENSE)
