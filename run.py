"""
Lanzador de MoodleAssist-AI desde la raíz del proyecto.
Permite ejecutar directamente: python run.py
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Asegurar que moodle_assistant sea importable
workspace_dir = Path(__file__).resolve().parent
if str(workspace_dir) not in sys.path:
    sys.path.insert(0, str(workspace_dir))

from moodle_assistant.main import main

if __name__ == "__main__":
    main()

