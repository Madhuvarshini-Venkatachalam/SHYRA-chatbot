"""
Custom AI Chatbot with Memory — Web App Backend (Gemini version)
DecodeLabs Industrial Training Kit — Project 1

Flask server that:
  - Serves a chat UI
  - Maintains an in-memory conversation history (per server session, using
    a simple global list for single-user local demo purposes)
  - Connects to Gemini via the official Google GenAI SDK
  - Validates input before sending (blocks empty/whitespace payloads)
  - Applies a sliding-window (FIFO) truncation to avoid context overflow
  - Grounds responses with live Google Search when current info is needed

Setup:
  pip install flask google-genai
  export GEMINI_API_KEY="your-key-here"
  python app.py

Then open: http://127.0.0.1:5000
"""

import os
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = "gemini-2.5-flash"
MAX_TOKENS = 1024
SYSTEM_PROMPT = (
    "You are Shyra, a friendly, warm, and cheerful AI companion shaped like "
    "a cute girl robot. You speak in a sweet, helpful, slightly playful tone "
    "(occasional soft emoji like ✨ or 💜 is okay, but don't overdo it). "
    "Keep replies clear and concise. You have access to live web search — "
    "use it whenever a question involves current events, recent dates, "
    "people currently holding a position, prices, or anything that could "
    "have changed since you were trained. Don't mention the tool by name, "
    "just answer naturally with up-to-date info."
)
MAX_HISTORY_MESSAGES = 20  # sliding window cap (user+model entries)

# Web search grounding tool — lets Shyra pull live info from Google Search
search_tool = types.Tool(google_search=types.GoogleSearch())

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        'Set the GEMINI_API_KEY environment variable first.\n'
        '  export GEMINI_API_KEY="your-key-here"'
    )

client = genai.Client(api_key=api_key)
app = Flask(__name__)

# In-memory conversation history (H_t-1). Resets when the server restarts.
history = []


def validate_input(text: str) -> bool:
    """Structural Validation Gate: block empty / whitespace-only payloads."""
    return bool(text and text.strip())


def apply_sliding_window(hist: list, max_messages: int) -> list:
    """FIFO pruning: drop oldest messages once history exceeds the cap."""
    if len(hist) > max_messages:
        overflow = len(hist) - max_messages
        if overflow % 2 != 0:
            overflow += 1
        return hist[overflow:]
    return hist


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    global history

    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "")

    if not validate_input(user_message):
        return jsonify({"error": "Empty message ignored."}), 400

    # Ingest & append user turn (Gemini schema: role + parts list)
    history.append({"role": "user", "parts": [{"text": user_message}]})
    history = apply_sliding_window(history, MAX_HISTORY_MESSAGES)

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=MAX_TOKENS,
                tools=[search_tool],
            ),
        )
        reply = response.text

        # Detect whether the model actually used search grounding, and
        # collect source links if so (for transparency in the UI).
        used_search = False
        sources = []
        candidate = response.candidates[0] if response.candidates else None
        grounding = getattr(candidate, "grounding_metadata", None) if candidate else None
        if grounding and getattr(grounding, "grounding_chunks", None):
            used_search = True
            for chunk in grounding.grounding_chunks:
                web = getattr(chunk, "web", None)
                if web and web.uri:
                    sources.append({"title": web.title or web.uri, "uri": web.uri})

    except Exception as e:
        return jsonify({"error": f"API error: {e}"}), 500

    # Append model response (Gemini uses role "model", not "assistant")
    history.append({"role": "model", "parts": [{"text": reply}]})
    history = apply_sliding_window(history, MAX_HISTORY_MESSAGES)

    return jsonify({"reply": reply, "used_search": used_search, "sources": sources})


@app.route("/api/reset", methods=["POST"])
def reset():
    global history
    history = []
    return jsonify({"status": "history cleared"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
