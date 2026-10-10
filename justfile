# GenAI Graph — project justfile
# Shared recipes imported from genai-tk/tk.just.
# Project-specific recipes defined here.

set dotenv-load
set dotenv-path := "~/.env"
set shell := ["bash", "-euc"]
set positional-arguments

pkg_name := "genai_graph"
app := "genai-graph"
image_version := "0.2a"
aws_region := "eu-west-1"
aws_account_id := "909658914353"
streamlit_entry := "genai_graph/main/streamlit.py"
modal_entry := "genai_graph/main/modal_app.py"
dev_pythonpath := "../genai-tk:.:${PWD}"

# Import shared genai-tk recipes
import? '../genai-tk/tk.just'

# Import deployment modules
mod docker 'deploy/docker.just'
mod modal 'deploy/modal.just'
mod prefect 'deploy/prefect.just'

# List available recipes
default:
    @just --list --unsorted

# ─── Web Applications ───────────────────────────────────────────────────────

[doc('Launch FastAPI server locally')]
fast-api:
    PYTHONPATH={{ dev_pythonpath }} uv run uvicorn genai_graph.main.fastapi_app:app --reload

[doc('Launch LangServe app')]
langserve:
    PYTHONPATH={{ dev_pythonpath }} uv run python genai_graph/main/langserve_app.py

# ─── Testing ────────────────────────────────────────────────────────────────

[doc('Quick smoke-test: call a fake LLM via the CLI')]
test-install:
    #!/usr/bin/env bash
    set -euo pipefail
    echo -e "\033[3m\033[36mExpected output: 'tell me a joke on bears'\033[0m"
    PYTHONPATH={{ dev_pythonpath }} uv run cli core llm -i "tell me a joke on bears" -m parrot_local@fake

# ─── Graph Tools ────────────────────────────────────────────────────────────

[doc('Start Ladybug DB explorer at http://localhost:8000')]
ladybug-explorer:
    docker run --rm -p 8000:8000 \
        -v /home/tcl/kuzu:/database \
        -e KUZU_FILE=ekg_database.db \
        kuzudb/explorer:latest &
    xdg-open http://localhost:8000

[doc('Generate BAML Python client from baml_src')]
baml-generate:
    uv run baml-cli generate --from ./genai_graph/ekg/baml_src

# ─── Infrastructure ─────────────────────────────────────────────────────────

[doc('Start Postgres + pgvector container')]
postgres:
    docker rm -f pgvector-container 2>/dev/null || true
    docker run -d --name pgvector-container \
        -e POSTGRES_USER=${POSTGRES_USER} \
        -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD} \
        -e POSTGRES_DB=ekg \
        -p 5432:5432 \
        -v /home/tcl/pgvector-data:/var/lib/postgresql/data \
        pgvector/pgvector:pg17

[doc('Start Chromium container (UI at http://localhost:3000)')]
chrome:
    docker rm -f chromium 2>/dev/null || true
    docker run -d --name=chromium \
        --security-opt seccomp=unconfined \
        -e PUID=1000 -e PGID=1000 -e TZ=Europe/Paris \
        -p 3000:3000 -p 3001:3001 \
        -v /home/tcl/.chromiun:/config \
        --shm-size="1gb" --restart unless-stopped \
        lscr.io/linuxserver/chromium:latest
    xdg-open localhost:3000
