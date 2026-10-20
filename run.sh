#!/usr/bin/env bash
# Shortcuts for this project (make isn't installed on the machine it was built on).
set -euo pipefail
cd "$(dirname "$0")"

langfuse_env() {
  # local-only secrets, generated once; the API reads the project keys from here
  if [ ! -f langfuse/.env ]; then
    cat > langfuse/.env <<ENV
SALT=$(openssl rand -hex 16)
ENCRYPTION_KEY=$(openssl rand -hex 32)
NEXTAUTH_SECRET=$(openssl rand -hex 16)
LANGFUSE_INIT_ORG_ID=aml
LANGFUSE_INIT_ORG_NAME=aml-graphrag
LANGFUSE_INIT_PROJECT_ID=aml-graphrag
LANGFUSE_INIT_PROJECT_NAME=aml-graphrag
LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-$(openssl rand -hex 8)
LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-$(openssl rand -hex 16)
LANGFUSE_INIT_USER_EMAIL=analyst@aml-graphrag.localhost
LANGFUSE_INIT_USER_NAME=analyst
LANGFUSE_INIT_USER_PASSWORD=$(openssl rand -hex 8)
ENV
    echo "created langfuse/.env (login: analyst@aml-graphrag.localhost, password inside)"
  fi
}

with_langfuse() {
  if [ -f langfuse/.env ]; then
    set -a; . langfuse/.env; set +a
    export LANGFUSE_PUBLIC_KEY=$LANGFUSE_INIT_PROJECT_PUBLIC_KEY LANGFUSE_SECRET_KEY=$LANGFUSE_INIT_PROJECT_SECRET_KEY
    export LANGFUSE_HOST=http://localhost:3000
  fi
  "$@"
}

case "${1:-help}" in
  test)      uv run pytest -q ;;
  ollama)    OLLAMA_NUM_PARALLEL=3 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve ;;
  corpus)    uv run scripts/fetch_corpus.py ;;
  index)     uv run scripts/extract.py && uv run scripts/build_index.py ;;
  gnn)       uv run --group gnn scripts/train_gnn.py && uv run --group gnn scripts/explain_alerts.py ;;
  evaluate)  uv run scripts/make_questions.py && uv run scripts/answer.py && uv run scripts/judge.py
             uv run scripts/evaluate_analyst.py answer && uv run scripts/evaluate_analyst.py judge
             uv run scripts/evaluate_analyst.py report ;;
  langfuse)  langfuse_env; docker compose -f langfuse/docker-compose.yml --env-file langfuse/.env up -d
             echo "Langfuse on http://localhost:3000" ;;
  langfuse-down) docker compose -f langfuse/docker-compose.yml --env-file langfuse/.env down ;;
  api)       with_langfuse uv run --group mlops uvicorn aml_graphrag.entrypoints.api:app --port 8000 ;;
  web)       cd web && PATH=$HOME/.local/opt/node/bin:$PATH npm start ;;
  *)         echo "usage: ./run.sh test|ollama|corpus|index|gnn|evaluate|langfuse|langfuse-down|api|web" ;;
esac
