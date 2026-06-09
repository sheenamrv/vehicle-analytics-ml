import subprocess
import threading
import time
from pathlib import Path

import requests
import webview

streamlit_process = None

## to build run pyinstaller --onedir --windowed run.py
def start_streamlit():
    global streamlit_process

    app_path = Path(__file__).parent / "app.py"

    streamlit_process = subprocess.Popen(
        [
            "streamlit",
            "run",
            str(app_path),
            "--server.headless=true",
            "--browser.gatherUsageStats=false"
        ]
    )


def wait_for_streamlit():
    while True:
        try:
            requests.get("http://localhost:8501")
            return
        except:
            time.sleep(1)


if __name__ == "__main__":

    threading.Thread(
        target=start_streamlit,
        daemon=True
    ).start()

    wait_for_streamlit()

    window = webview.create_window(
        "Predictive Analytics Dashboard",
        "http://localhost:8501",
        maximized=True,
        resizable=True
    )

    webview.start()

    if streamlit_process:
        streamlit_process.kill()