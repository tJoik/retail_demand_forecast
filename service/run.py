"""Запуск сервиса: python service/run.py"""
import os
import sys
from pathlib import Path

# Убедимся что CWD = корень проекта (retail_demand_forecast/)
os.chdir(Path(__file__).parent.parent)
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from app import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
