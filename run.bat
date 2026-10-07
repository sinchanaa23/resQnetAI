@echo off
cd /d "%~dp0"
python -m pip install -q -r requirements.txt
start "" /min cmd /c "timeout /t 10 >/dev/null & start http://localhost:8000"
python -m uvicorn backend.app:app --port 8000
