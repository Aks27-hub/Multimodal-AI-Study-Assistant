"""
app.py
════════════════════════════════════════════════════════════════
Handwriting OCR pipeline using Qwen2.5-VL + Gemini,
with a custom HTML UI served locally from Flask.

Run locally:
    python app.py

Environment:
    QWEN_MODEL_PATH   Hugging Face model id or local model path
    GEMINI_API_KEY    Gemini API key for correction / generation

Default model:
    Qwen/Qwen2.5-VL-7B-Instruct
    Set QWEN_MODEL_PATH to 2B or 3B if you want a different size.
════════════════════════════════════════════════════════════════
"""

import json
import os
import uuid
import tempfile
import traceback
from pathlib import Path
from flask import Flask, jsonify, request, render_template
from backend import *
from rag_engine import *

GEMINI_MODEL = "gemini-3.5-flash"
NOTES_FILE = Path(os.getenv("INKWELL_NOTES_FILE", "inkwell_notes.json"))

flask_app = Flask(__name__)

# qwen_model = get_qwen()
# qwen_processor = get_processor()
_notes_state = []

def load_notes() -> list[dict]:
    if not NOTES_FILE.exists():
        return []

    try:
        data = json.loads(NOTES_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass

    return []


def save_notes(notes: list[dict]) -> None:
    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    NOTES_FILE.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")


_notes_state = load_notes()


def strip_code_fences(text: str) -> str:
    return text.strip().replace("```json", "").replace("```", "").strip()


def safe_text(value) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def all_notes_text(notes_state: list[dict]) -> str:
    return "\n\n---\n\n".join(note.get("text", "") for note in notes_state if note.get("text"))


def generate_flashcards(notes_state: list[dict]) -> list[dict]:
    if not notes_state:
        return []

    client = get_gemini_client()
    if client is None:
        raise RuntimeError("Set GEMINI_API_KEY to generate flashcards.")

    prompt = f"""Based on these notes, generate 6-8 flashcards.
Return ONLY a JSON array of objects with keys "front" and "back". No markdown.

Notes:
{all_notes_text(notes_state)}"""

    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    cards = json.loads(strip_code_fences(response.text))
    if not isinstance(cards, list):
        raise ValueError("Flashcard response was not a JSON array.")
    return cards


def generate_quiz(notes_state: list[dict]) -> list[dict]:
    if not notes_state:
        return []

    client = get_gemini_client()
    if client is None:
        raise RuntimeError("Set GEMINI_API_KEY to generate quiz questions.")

    prompt = f"""Based on these notes, generate 5 multiple-choice questions.
Return ONLY a JSON array with keys: "question", "options" (4 strings), "answer" (0-3), "explanation".
No markdown.

Notes:
{all_notes_text(notes_state)}"""

    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    quiz = json.loads(strip_code_fences(response.text))
    if not isinstance(quiz, list):
        raise ValueError("Quiz response was not a JSON array.")
    return quiz


def generate_misconceptions(notes_state: list[dict]) -> list[dict]:
    if not notes_state:
        return []

    client = get_gemini_client()
    if client is None:
        raise RuntimeError("Set GEMINI_API_KEY to generate misconceptions.")

    prompt = f"""You are an expert teacher. Analyse these student notes for:
1. Common misconceptions or errors in understanding
2. Missing key concepts
3. Clarifications needed for ambiguous statements

Return ONLY a JSON array with keys:
  "type": "misconception"|"missing"|"clarification"
  "original": exact text from notes or ""
  "issue": what is wrong or missing
  "correction": the correct explanation

Notes:
{all_notes_text(notes_state)}"""

    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    items = json.loads(strip_code_fences(response.text))
    if not isinstance(items, list):
        raise ValueError("Misconception response was not a JSON array.")
    return items


def generate_mindmap(notes_state: list[dict]) -> dict:
    if not notes_state:
        return {}

    client = get_gemini_client()
    if client is None:
        raise RuntimeError("Set GEMINI_API_KEY to generate a mind map.")

    prompt = f"""Based on these notes, extract a central topic and 4-7 main concepts with 2-4 sub-points each.
Return ONLY JSON: {{"center": "string", "nodes": [{{"label": "string", "children": ["string"]}}]}}
No markdown.

Notes:
{all_notes_text(notes_state)}"""

    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    data = json.loads(strip_code_fences(response.text))
    if not isinstance(data, dict):
        raise ValueError("Mind map response was not a JSON object.")
    return data


def render_notes_html(notes: list[dict]) -> str:
    if not notes:
        return """<div class="empty-state">
          <div class="empty-emoji">📝</div>
          <div class="empty-title">No notes yet</div>
          <div class="empty-copy">Extract text from an image and save it here.</div>
        </div>"""

    cards = []
    accent_colors = ["#c84b2f", "#2f6bc8", "#c8962f", "#2d7a4f", "#8b2fc8"]
    for index, note in enumerate(notes):
        color = accent_colors[index % len(accent_colors)]
        preview = safe_text((note.get("text", "")[:180] + "…") if len(note.get("text", "")) > 180 else note.get("text", ""))
        title = safe_text(note.get("title", "Untitled"))
        cards.append(
            f'''<div class="note-card" style="border-left-color:{color};">
              <div class="note-meta">{safe_text(note.get("date", ""))}</div>
              <div class="note-title">{title}</div>
              <div class="note-preview">{preview}</div>
            </div>'''
        )
    return "".join(cards)


def render_flashcard_html(cards, idx):
    if not cards:
        return """<div class="empty-state">
          <div class="empty-emoji">🃏</div>
          <div class="empty-title">No flashcards yet</div>
          <div class="empty-copy">Generate from your saved notes.</div>
        </div>"""

    card = cards[idx]
    total = len(cards)
    return f"""
    <div class="fc-wrap">
      <div class="fc" onclick="this.classList.toggle('flipped')">
        <div class="fc-face fc-front">
          <div class="fc-label">Question</div>
          <div class="fc-text">{safe_text(card.get('front', ''))}</div>
          <div class="fc-hint">Click to flip</div>
        </div>
        <div class="fc-face fc-back">
          <div class="fc-label">Answer</div>
          <div class="fc-text">{safe_text(card.get('back', ''))}</div>
        </div>
      </div>
      <div class="fc-counter">{idx + 1} / {total}</div>
    </div>"""


def render_quiz_html(quiz, q_idx, answered):
    if not quiz:
        return """<div class="empty-state">
          <div class="empty-emoji">❓</div>
          <div class="empty-title">No quiz yet</div>
          <div class="empty-copy">Generate from your saved notes.</div>
        </div>"""

    total = len(quiz)
    q = quiz[q_idx]
    user_ans = answered.get(q_idx)
    dots = []
    for index in range(total):
        if index in answered:
            color = "#2d7a4f"
        elif index == q_idx:
            color = "#1a1714"
        else:
            color = "#d4ccbb"
        dots.append(f'<div style="background:{color};"></div>')

    opts_html = []
    for index, opt in enumerate(q.get("options", [])):
        if user_ans is None:
            style_class = "option neutral"
            onclick = f"onclick=\"answerQuiz({index})\""
        elif index == q.get("answer"):
            style_class = "option correct"
            onclick = ""
        elif index == user_ans:
            style_class = "option wrong"
            onclick = ""
        else:
            style_class = "option neutral"
            onclick = ""
        opts_html.append(f'<button class="{style_class}" {onclick}>{safe_text(opt)}</button>')

    explanation = ""
    if user_ans is not None and q.get("explanation"):
        explanation = f'<div class="quiz-explanation">💡 {safe_text(q["explanation"])}</div>'

    return f"""
    <div class="quiz-shell">
      <div class="progress-dots">{''.join(dots)}</div>
      <div class="quiz-card">
        <div class="quiz-question">{safe_text(q.get('question', ''))}</div>
        <div class="quiz-options">{''.join(opts_html)}</div>
        {explanation}
      </div>
      <div class="quiz-counter">Question {q_idx + 1} of {total}</div>
    </div>"""


def render_misconceptions_html(items):
    if not items:
        return """<div class="empty-state">
          <div class="empty-emoji">✅</div>
          <div class="empty-title">All clear!</div>
          <div class="empty-copy">No misconceptions found in your notes.</div>
        </div>"""

    cfg = {
        "misconception": ("#c84b2f", "❌ Misconception"),
        "missing": ("#c8962f", "⚠️ Missing Concept"),
        "clarification": ("#2f6bc8", "💡 Clarification"),
    }

    html = []
    for item in items:
        color, label = cfg.get(item.get("type", ""), ("#7a7168", "ℹ️ Note"))
        original = item.get("original", "")
        original_html = f'<div class="analysis-source">From notes: "{safe_text(original)}"</div>' if original else ""
        html.append(
            f'''<div class="analysis-card" style="border-left-color:{color};">
              <div class="analysis-label" style="color:{color};">{label}</div>
              <div class="analysis-issue">{safe_text(item.get("issue", ""))}</div>
              {original_html}
              <div class="analysis-correction">{safe_text(item.get("correction", ""))}</div>
            </div>'''
        )
    return "".join(html)


def render_mindmap_html(data):
    if not data:
        return """<div class="empty-state">
          <div class="empty-emoji">🗺️</div>
          <div class="empty-title">No mind map yet</div>
          <div class="empty-copy">Generate from your saved notes.</div>
        </div>"""

    import math

    width, height = 900, 520
    center_x, center_y = width // 2, height // 2
    radius_1, radius_2 = 165, 290
    nodes = data.get("nodes", [])
    step = (2 * math.pi) / max(len(nodes), 1)
    colors = ["#c84b2f", "#2f6bc8", "#c8962f", "#2d7a4f", "#8b2fc8", "#c82f8b", "#2fc8aa"]

    svg_lines = []
    svg_nodes = []
    svg_nodes.append(
        f'<rect x="{center_x - 74}" y="{center_y - 22}" width="148" height="44" rx="10" fill="#1a1714"/>'
        f'<text x="{center_x}" y="{center_y + 5}" text-anchor="middle" fill="#f5f0e8" '
        f'font-family="Playfair Display,serif" font-size="13" font-weight="700">'
        f'{safe_text(data.get("center", ""))}</text>'
    )

    for index, node in enumerate(nodes):
        angle = step * index - math.pi / 2
        node_x = center_x + radius_1 * math.cos(angle)
        node_y = center_y + radius_1 * math.sin(angle)
        color = colors[index % len(colors)]
        label = str(node.get("label", ""))
        width_box = max(84, len(label) * 7 + 24)

        svg_lines.append(
            f'<line x1="{center_x}" y1="{center_y}" x2="{node_x:.1f}" y2="{node_y:.1f}" '
            f'stroke="{color}" stroke-width="2" opacity="0.4"/>'
        )
        svg_nodes.append(
            f'<rect x="{node_x - width_box / 2:.1f}" y="{node_y - 16:.1f}" width="{width_box}" height="32" '
            f'rx="8" fill="{color}" opacity="0.92"/>'
            f'<text x="{node_x:.1f}" y="{node_y + 5:.1f}" text-anchor="middle" fill="white" '
            f'font-family="DM Sans,sans-serif" font-size="11" font-weight="500">'
            f'{safe_text(label)}</text>'
        )

        children = node.get("children", [])
        spread = math.pi * 0.55
        for child_index, child in enumerate(children):
            if len(children) == 1:
                child_angle = angle
            else:
                child_angle = angle - spread / 2 + (spread / (len(children) - 1)) * child_index
            child_x = center_x + radius_2 * math.cos(child_angle)
            child_y = center_y + radius_2 * math.sin(child_angle)
            child_width = max(72, len(str(child)) * 6.5 + 20)

            svg_lines.append(
                f'<line x1="{node_x:.1f}" y1="{node_y:.1f}" x2="{child_x:.1f}" y2="{child_y:.1f}" '
                f'stroke="{color}" stroke-width="1.2" opacity="0.22" stroke-dasharray="4 3"/>'
            )
            svg_nodes.append(
                f'<rect x="{child_x - child_width / 2:.1f}" y="{child_y - 13:.1f}" width="{child_width:.1f}" height="26" '
                f'rx="6" fill="{color}" opacity="0.14" stroke="{color}" stroke-width="1" stroke-opacity="0.35"/>'
                f'<text x="{child_x:.1f}" y="{child_y + 4:.1f}" text-anchor="middle" fill="#1a1714" '
                f'font-family="DM Sans,sans-serif" font-size="10">{safe_text(child)}</text>'
            )

    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;border-radius:12px;border:1px solid #d4ccbb;">'
        f'<rect width="{width}" height="{height}" fill="#faf7f2"/>'
        f'{"".join(svg_lines)}{"".join(svg_nodes)}</svg>'
    )


@flask_app.get("/")
def index():
    return render_template("index.html", initial_notes=_notes_state)


@flask_app.get("/api/notes")
def api_notes():
    return jsonify({"notes": _notes_state})


@flask_app.post("/api/ocr")
def api_ocr():
    uploaded = request.files.get("image")
    if uploaded is None:
        return jsonify({"error": "Please upload an image."}), 400

    tmp_path = None
    try:
        client = get_gemini_client()
        suffix = Path(uploaded.filename or "image.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            uploaded.save(tmp.name)
            tmp_path = tmp.name

        qwen_model = get_qwen()
        qwen_processor = get_processor()
        raw_lines = transcribe_with_qwen(qwen_model, qwen_processor, tmp_path)
        corrected_lines = correct_with_llm(raw_lines)
        text = "\n".join(corrected_lines)
        stats = f"✓ {len(corrected_lines)} lines · {len(text.split())} words · {'Gemini-corrected' if client else 'local OCR only'}"
        return jsonify({"text": text, "stats": stats})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@flask_app.post("/api/notes")
def api_save_note():
    payload = request.get_json(force=True, silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text or text.startswith("⚠️") or text.startswith("❌"):
        return jsonify({"error": "Nothing to save."}), 400

    import datetime

    title = text.splitlines()[0][:60] or "Untitled"
    note_id = str(uuid.uuid4())
    note = {
        "id": note_id,
        "title": title,
        "text": text,
        "date": datetime.datetime.now().strftime("%b %d, %Y"),
    }
    _notes_state.insert(0, note)
    save_notes(_notes_state)

    # Automatically add to local session vector store
    index_session_note(note_id=note_id, text=text, metadata={"title": title})
    return jsonify({"notes": _notes_state, "note": note})


@flask_app.delete("/api/notes/<int:index>")
def api_delete_note(index: int):
    if index < 0 or index >= len(_notes_state):
        return jsonify({"error": "Note index out of range."}), 404

    # del _notes_state[index]
    rem_note = _notes_state.pop(index)
    if "id" in rem_note:
        session_collection.delete(ids=[rem_note["id"]])
    save_notes(_notes_state)
    return jsonify({"notes": _notes_state})


@flask_app.post("/api/notes/clear")
def api_clear_notes():
    _notes_state.clear()
    save_notes(_notes_state)
    session_collection.delete(ids=session_collection.get()['ids'])  # Clear local session DB
    return jsonify({"notes": _notes_state})


@flask_app.post("/api/flashcards")
def api_flashcards():
    try:
        return jsonify({"cards": generate_flashcards(_notes_state)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@flask_app.post("/api/quiz")
def api_quiz():
    try:
        return jsonify({"quiz": generate_quiz(_notes_state)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@flask_app.post("/api/misconceptions")
def api_misconceptions():
    try:
        return jsonify({"html": render_misconceptions_html(generate_misconceptions(_notes_state))})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@flask_app.post("/api/mindmap")
def api_mindmap():
    try:
        return jsonify({"html": render_mindmap_html(generate_mindmap(_notes_state))})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@flask_app.post("/api/rag_query")
def api_rag_query():
    payload = request.get_json(force=True, silent=True) or {}
    user_query = payload.get("query", "").strip()
    
    if not user_query:
        return jsonify({"error": "Query cannot be empty."}), 400

    # 1. Local-first Retrieval
    docs, source_db = retrieve_relevant_context(user_query, threshold=0.8)
    
    if not docs:
        return jsonify({"answer": "No relevant notes found in local or global databases.", "source": "none"})

    context_str = "\n\n---\n\n".join(docs)

    # 2. Synthesis
    client = get_gemini_client()
    if client is None:
        return jsonify({"error": "GEMINI_API_KEY is missing."}), 500

    prompt = f"""Use the provided context to answer the user's question accurately.

Context ({source_db} database):
{context_str}

Question: {user_query}
Answer:"""

    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    
    return jsonify({
        "answer": response.text.strip(),
        "source": source_db,
        "retrieved_chunks": docs
    })

def launch(host="127.0.0.1", port=7860, debug=False, **kwargs):
    """Launch the local HTML app."""
    flask_app.run(host=host, port=port, debug=debug, **kwargs)


if __name__ == "__main__":
    launch()
