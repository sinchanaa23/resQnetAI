#!/usr/bin/env sh
cd "$(dirname "$0")"
python3 -m pip install -q -r requirements.txt
(sleep 6; (xdg-open http://localhost:8000 || open http://localhost:8000) >/dev/null 2>&1) &
python3 -m uvicorn backend.app:app --port 8000
