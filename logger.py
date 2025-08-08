# logger.py
import datetime
import os

LOG_FILE = "logs/training.log"
os.makedirs("logs", exist_ok=True)

def log(message, prefix="INFO"):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    formatted = f"{timestamp} [{prefix}] {message}"
    with open(LOG_FILE, "a") as f:
        f.write(formatted + "\n")
        f.write("=" * 80 + "\n")
    print(formatted)
    print("=" * 80)