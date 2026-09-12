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

import os
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
import anthropic
from dotenv import load_dotenv

load_dotenv()

APP_DIR = Path(__file__).parent
WEB_DIR = APP_DIR / "web"

app = Flask(__name__, static_folder=None)

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
    return send_from_directory(WEB_DIR, filename)


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

    messages = list(history) + [{"role": "user", "content": phrase}]
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    reply = next(
        (block.text for block in response.content if getattr(block, "type", None) == "text"),
        "",
    )
    return jsonify({"reply": reply})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Serving on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
