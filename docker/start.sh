#!/bin/sh
# serve      : language model + API (:8000) + dashboard (:8501)
# benchmark  : timing / memory benchmark, prints a report (run with --network none to prove it is offline)
# anything else is run as a command
set -e
cd /app

case "${1:-serve}" in
  serve)
    # llama.cpp server for Qwen2.5-3B (the API and dashboard wait for it and fall back to templates if it is down)
    llama-server -m models/qwen2.5-3b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8081 \
                 -t "${SHELFSENSE_THREADS:-4}" -c 2048 --no-webui > /tmp/llama.log 2>&1 &
    uvicorn shelfsense.api.app:app --host 0.0.0.0 --port 8000 &
    exec streamlit run shelfsense/dashboard/app.py --server.address 0.0.0.0 --server.port 8501 \
         --server.headless true
    ;;
  benchmark)
    shift
    exec python -m shelfsense.eval.benchmark "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
