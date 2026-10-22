"""
server.py
Serves the mobile web version (web/) of the silent-speech-to-AI assistant,
and proxies AI requests to Claude so your API key stays on this laptop —
it's never sent to or visible in the phone's browser.

Setup: same .env file as ai_chat.py (ANTHROPIC_API_KEY=...).

Run:
    python server.py
Then see README_IPHONE.md for how to reach it from your phone.
"""

import csv
import json
import mimetypes
import os
import re
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory

# .webmanifest is not in Python's default table; without this the phone gets
# it as application/octet-stream and ignores it.
mimetypes.add_type("application/manifest+json", ".webmanifest")
import anthropic
from dotenv import load_dotenv

load_dotenv()

APP_DIR = Path(__file__).parent
WEB_DIR = APP_DIR / "web"
LOG_DIR = APP_DIR / "logs"
TEST_DIR = LOG_DIR / "tests"
TEST_CSV = LOG_DIR / "test-results.csv"

app = Flask(__name__, static_folder=None)
# Test uploads carry raw mouth data (10 tries); cap size so nobody can fill the disk.
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError(
        "No ANTHROPIC_API_KEY found. Run setup_key.py first (same .env "
        "file ai_chat.py uses)."
    )
client = anthropic.Anthropic(api_key=api_key)

MODEL = "claude-sonnet-5"
SYSTEM_PROMPT = (
    "You are a voice assistant responding to a short spoken question. "
    "Answer in 1-3 sentences, conversationally, since your reply will be "
    "read aloud."
)


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    response = send_from_directory(WEB_DIR, filename)
    if filename == "sw.js":
        # The service worker must never be served stale, or the phone keeps
        # running an old copy of the app after you change it.
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Service-Worker-Allowed"] = "/"
    return response


@app.route("/models/face_landmarker.task")
def model_file():
    return send_from_directory(APP_DIR, "face_landmarker.task")


@app.route("/ask", methods=["POST"])
def ask():
    body = request.get_json(force=True)
    phrase = (body or {}).get("phrase", "")
    history = (body or {}).get("history", [])

    if not phrase:
        return jsonify({"error": "no phrase provided"}), 400

    # The phone sends its local time; fall back to the laptop's clock for an
    # older cached app. Capped so a bad request can't stuff the prompt.
    now = str((body or {}).get("now") or "")[:80] or datetime.now().strftime(
        "%A %d %B %Y at %H:%M"
    )
    system = f"{SYSTEM_PROMPT} The user's current local date and time is {now}."

    messages = list(history) + [{"role": "user", "content": phrase}]
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=system,
        messages=messages,
    )
    reply = next(
        (block.text for block in response.content if getattr(block, "type", None) == "text"),
        "",
    )
    return jsonify({"reply": reply})


@app.route("/test-log", methods=["POST"])
def test_log():
    """Saves one 10-try test from the phone's Test tab.

    Full detail (every try, plus raw mouth data when sent) goes to
    logs/tests/<date>_<phrase>_<id>.json. One summary line per test goes to
    logs/test-results.csv, which opens in Excel.
    """
    body = request.get_json(force=True, silent=True) or {}
    result = body.get("result")
    if not isinstance(result, dict) or not result.get("phrase"):
        return jsonify({"error": "no test result"}), 400

    test_id = re.sub(r"[^0-9]", "", str(result.get("id", "")))[:16] or "0"
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    if any(TEST_DIR.glob(f"*_{test_id}.json")):
        return jsonify({"ok": True, "duplicate": True})   # phone re-sent it

    phrase = str(result["phrase"])[:80]
    slug = re.sub(r"[^a-z0-9]+", "-", phrase.lower()).strip("-")[:40] or "phrase"
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    with open(TEST_DIR / f"{stamp}_{slug}_{test_id}.json", "w", encoding="utf-8") as f:
        json.dump(body, f)

    settings = result.get("settings") or {}
    new_file = not TEST_CSV.exists()
    with open(TEST_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["date", "phrase", "score", "out_of", "working", "light",
                        "clips_trained", "phrases_trained",
                        "movement_sensitivity", "match_strictness"])
        w.writerow([
            str(result.get("date", ""))[:19].replace("T", " "),
            phrase,
            result.get("score", ""),
            result.get("total", ""),
            "yes" if result.get("passed") else "no",
            result.get("light", ""),
            result.get("clips", ""),
            result.get("phrasesTrained", ""),
            settings.get("movementThreshold", ""),
            # v2 matcher sends acceptFactor; older app copies send maxDistance.
            settings.get("acceptFactor", settings.get("maxDistance", "")),
        ])
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Serving on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
