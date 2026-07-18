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
import tempfile
import traceback
from pathlib import Path
from flask import Flask, jsonify, request
from backend import *

GEMINI_MODEL = "gemini-2.5-flash"
NOTES_FILE = Path(os.getenv("INKWELL_NOTES_FILE", "inkwell_notes.json"))

flask_app = Flask(__name__)

qwen_model = get_qwen()
qwen_processor = get_processor()
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


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Inkwell — Handwriting to Knowledge</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@400;500;700&family=DM+Mono:wght@400;500&display=swap');

    :root {
      --ink: #1a1714;
      --paper: #f5f0e8;
      --cream: #ede7d9;
      --card: #faf7f2;
      --border: #d4ccbb;
      --muted: #7a7168;
      --accent: #c84b2f;
      --green: #2d7a4f;
      --shadow: 0 20px 40px rgba(26, 23, 20, 0.08);
    }

    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(200, 75, 47, 0.08), transparent 30%),
        radial-gradient(circle at top right, rgba(47, 107, 200, 0.06), transparent 28%),
        linear-gradient(180deg, #f8f4ee 0%, #f2ece3 100%);
      color: var(--ink);
      font-family: 'DM Sans', sans-serif;
    }

    .page { max-width: 1440px; margin: 0 auto; padding: 24px; }
    .topbar {
      position: sticky; top: 12px; z-index: 10;
      display: flex; justify-content: space-between; align-items: center; gap: 16px;
      padding: 18px 20px; margin-bottom: 22px;
      background: rgba(250, 247, 242, 0.86); backdrop-filter: blur(14px);
      border: 1px solid rgba(212, 204, 187, 0.9); border-radius: 20px; box-shadow: var(--shadow);
    }
    .brand { display: flex; align-items: center; gap: 14px; }
    .brand-mark {
      width: 42px; height: 42px; border-radius: 12px; background: var(--ink);
      display: grid; place-items: center; flex: 0 0 auto;
    }
    .brand-title { font-family: 'Playfair Display', serif; font-weight: 700; font-size: 1.45rem; line-height: 1; margin: 0 0 4px; letter-spacing: -0.02em; }
    .brand-sub { margin: 0; color: var(--muted); font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; }
    .toolbar { display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }
    .toolbar a {
      text-decoration: none; color: var(--ink); font-size: 0.82rem; border: 1px solid var(--border);
      background: var(--card); padding: 9px 12px; border-radius: 999px; transition: 0.15s ease;
    }
    .toolbar a:hover { transform: translateY(-1px); border-color: var(--ink); }
    .hero { display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 20px; margin-bottom: 20px; }
    .panel { background: rgba(250, 247, 242, 0.94); border: 1px solid var(--border); border-radius: 22px; box-shadow: var(--shadow); overflow: hidden; }
    .panel-inner { padding: 22px; }
    .panel-head { margin-bottom: 18px; }
    .eyebrow { font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
    .section-title { margin: 0; font-family: 'Playfair Display', serif; font-size: 1.55rem; line-height: 1.1; letter-spacing: -0.02em; }
    .section-sub { margin: 8px 0 0; color: var(--muted); font-size: 0.92rem; line-height: 1.5; }
    .status-bar {
      border: 1px solid var(--border); background: var(--cream); color: var(--muted); border-radius: 12px;
      padding: 10px 12px; font-family: 'DM Mono', monospace; font-size: 0.8rem; margin-bottom: 12px;
      min-height: 42px; display: flex; align-items: center;
    }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .stack { display: grid; gap: 20px; }
    .controls { display: flex; gap: 10px; flex-wrap: wrap; }
    .btn {
      appearance: none; border: 0; border-radius: 12px; padding: 11px 16px; font: inherit;
      font-weight: 600; cursor: pointer; transition: 0.16s ease;
    }
    .btn:hover { transform: translateY(-1px); }
    .btn-primary { background: var(--ink); color: var(--paper); }
    .btn-primary:hover { background: #2c2925; }
    .btn-secondary { background: transparent; color: var(--ink); border: 1px solid var(--border); }
    .btn-secondary:hover { background: var(--cream); }
    .field {
      width: 100%; border: 1px solid var(--border); border-radius: 14px; background: #fffdfa; color: var(--ink);
      padding: 14px; font: inherit; outline: none;
    }
    .field:focus { border-color: var(--ink); }
    textarea.field { min-height: 210px; resize: vertical; line-height: 1.65; }
    .dropzone { border: 1.5px dashed var(--border); background: var(--card); border-radius: 18px; padding: 18px; display: grid; gap: 14px; }
    .notes-list, .analysis-list { display: grid; gap: 14px; }
    .note-card, .analysis-card { background: var(--card); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 16px; padding: 16px 16px 16px 18px; }
    .note-meta { font-size: 0.72rem; color: var(--muted); font-family: 'DM Mono', monospace; margin-bottom: 6px; }
    .note-title { margin: 0 0 8px; font-family: 'Playfair Display', serif; font-size: 1.02rem; line-height: 1.35; }
    .note-preview { color: var(--muted); font-size: 0.86rem; line-height: 1.6; white-space: pre-wrap; }
    .note-actions { margin-top: 10px; }
    .mini { padding: 8px 11px; border-radius: 10px; font-size: 0.8rem; }
    .empty-state { text-align: center; padding: 52px 18px; color: var(--muted); }
    .empty-emoji { font-size: 2.15rem; margin-bottom: 10px; }
    .empty-title { font-family: 'Playfair Display', serif; font-size: 1.05rem; color: var(--ink); margin-bottom: 6px; }
    .empty-copy { font-size: 0.86rem; }
    .fc-wrap { perspective: 1000px; max-width: 540px; margin: 0 auto; }
    .fc { position: relative; height: 250px; transform-style: preserve-3d; transition: transform 0.5s cubic-bezier(0.4, 0, 0.2, 1); cursor: pointer; }
    .fc.flipped { transform: rotateY(180deg); }
    .fc-face { position: absolute; inset: 0; backface-visibility: hidden; transform-style: preserve-3d; border-radius: 18px; display: flex; align-items: center; justify-content: center; padding: 30px; text-align: center; }
    .fc-front { background: #1a1714; color: #f5f0e8; border: 1px solid #2f2a25; }
    .fc-back { background: #faf7f2; color: #1a1714; border: 1px solid var(--border); transform: rotateY(180deg); }
    .fc-label { position: absolute; top: 16px; left: 18px; font-size: 0.62rem; letter-spacing: 0.14em; text-transform: uppercase; opacity: 0.55; font-family: 'DM Sans', sans-serif; }
    .fc-text { font-family: 'Playfair Display', serif; font-size: 1.18rem; line-height: 1.5; }
    .fc-hint { position: absolute; bottom: 16px; right: 18px; font-size: 0.72rem; opacity: 0.55; }
    .fc-counter, .quiz-counter { margin-top: 12px; text-align: center; color: var(--muted); font-size: 0.8rem; font-family: 'DM Mono', monospace; }
    .quiz-shell { max-width: 720px; margin: 0 auto; }
    .progress-dots { display: flex; gap: 5px; margin-bottom: 18px; }
    .progress-dots div { height: 4px; flex: 1; border-radius: 99px; }
    .quiz-card { background: var(--card); border: 1px solid var(--border); border-radius: 18px; padding: 22px 24px; }
    .quiz-question { font-family: 'Playfair Display', serif; font-size: 1.12rem; line-height: 1.5; margin-bottom: 16px; }
    .quiz-options { display: grid; gap: 10px; }
    .option { width: 100%; text-align: left; border-radius: 12px; padding: 12px 14px; border: 1.5px solid var(--border); background: #f5f0e8; cursor: pointer; font: inherit; transition: 0.14s ease; }
    .option:hover { transform: translateY(-1px); border-color: var(--ink); }
    .option.correct { background: #f0faf4; border-color: var(--green); color: var(--green); font-weight: 700; }
    .option.wrong { background: #fdf2f0; border-color: var(--accent); color: var(--accent); }
    .option.neutral { color: var(--muted); }
    .quiz-explanation { margin-top: 14px; padding: 12px 14px; border-radius: 12px; background: var(--cream); color: var(--muted); line-height: 1.6; font-size: 0.86rem; }
    .analysis-label { font-size: 0.67rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 8px; }
    .analysis-issue { font-size: 0.94rem; line-height: 1.6; margin-bottom: 8px; font-weight: 600; }
    .analysis-source { font-size: 0.78rem; color: var(--muted); font-style: italic; padding: 8px 10px; background: var(--cream); border-radius: 10px; margin-bottom: 8px; }
    .analysis-correction { font-size: 0.9rem; line-height: 1.65; color: var(--ink); }
    #mindmapArea svg { display: block; }
    .footer-note { margin-top: 18px; color: var(--muted); font-size: 0.82rem; text-align: center; }
    @media (max-width: 1100px) { .hero, .grid { grid-template-columns: 1fr; } .toolbar { justify-content: flex-start; } }
    @media (max-width: 720px) { .page { padding: 14px; } .topbar { padding: 16px; border-radius: 18px; } .panel-inner { padding: 18px; } .fc { height: 270px; } .fc-text { font-size: 1.05rem; } }
  </style>
</head>
<body>
  <div class="page">
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark">
          <svg width="20" height="20" fill="none" stroke="#f5f0e8" stroke-width="2" viewBox="0 0 24 24">
            <path d="M12 20h9M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4L16.5 3.5z"/>
          </svg>
        </div>
        <div>
          <h1 class="brand-title">Inkwell</h1>
          <p class="brand-sub">Qwen2.5-VL · Gemini · Handwriting to Knowledge</p>
        </div>
      </div>
      <nav class="toolbar">
        <a href="#scanSection">Scan</a>
        <a href="#notesSection">Notes</a>
        <a href="#flashcardsSection">Flashcards</a>
        <a href="#quizSection">Quiz</a>
        <a href="#analysisSection">Analysis</a>
        <a href="#mindmapSection">Mind Map</a>
      </nav>
    </header>

    <section class="hero">
      <div class="panel" id="scanSection">
        <div class="panel-inner">
          <div class="panel-head">
            <div class="eyebrow">Scan Handwriting</div>
            <h2 class="section-title">Turn a notebook page into editable text</h2>
            <p class="section-sub">Upload an image, let Qwen transcribe it locally, then optionally correct it with Gemini.</p>
          </div>
          <div class="dropzone">
            <input id="imageInput" class="field" type="file" accept="image/*" />
            <div class="controls">
              <button class="btn btn-primary" onclick="extractText()">Extract Text</button>
              <button class="btn btn-secondary" onclick="saveCurrentNote()">Save to Notes</button>
            </div>
            <div id="extractStatus" class="status-bar">Upload an image and click Extract Text.</div>
            <textarea id="extractedText" class="field" placeholder="Extracted handwriting will appear here..."></textarea>
            <div id="saveStatus" class="status-bar">Nothing saved yet.</div>
          </div>
        </div>
      </div>

      <div class="panel" id="notesSection">
        <div class="panel-inner">
          <div class="panel-head">
            <div class="eyebrow">Notes</div>
            <h2 class="section-title">Saved text is the source of every study tool</h2>
            <p class="section-sub">Refresh the list, clear everything, or delete individual notes.</p>
          </div>
          <div class="controls" style="margin-bottom:14px;">
            <button class="btn btn-secondary" onclick="refreshNotes()">Refresh</button>
            <button class="btn btn-secondary" onclick="clearNotes()">Clear All</button>
          </div>
          <div id="notesList" class="notes-list"></div>
        </div>
      </div>
    </section>

    <section class="grid">
      <div class="stack">
        <div class="panel" id="flashcardsSection">
          <div class="panel-inner">
            <div class="panel-head">
              <div class="eyebrow">Flashcards</div>
              <h2 class="section-title">Review by flipping cards</h2>
              <p class="section-sub">Cards are generated from your saved notes. Click a card to flip it.</p>
            </div>
            <div class="controls" style="justify-content:center;margin-bottom:14px;">
              <button class="btn btn-secondary" onclick="prevFlashcard()">← Prev</button>
              <button class="btn btn-primary" onclick="generateFlashcards()">Generate from Notes</button>
              <button class="btn btn-secondary" onclick="nextFlashcard()">Next →</button>
            </div>
            <div id="flashcardArea"></div>
          </div>
        </div>

        <div class="panel" id="analysisSection">
          <div class="panel-inner">
            <div class="panel-head">
              <div class="eyebrow">Misconceptions</div>
              <h2 class="section-title">Find gaps and misunderstandings</h2>
              <p class="section-sub">Gemini reviews the notes for missing concepts and wrong ideas.</p>
            </div>
            <div class="controls" style="margin-bottom:14px;">
              <button class="btn btn-primary" onclick="generateMisconceptions()">Analyse Notes</button>
            </div>
            <div id="misconceptionsArea" class="analysis-list"></div>
          </div>
        </div>
      </div>

      <div class="stack">
        <div class="panel" id="quizSection">
          <div class="panel-inner">
            <div class="panel-head">
              <div class="eyebrow">Quiz</div>
              <h2 class="section-title">Test yourself with multiple-choice questions</h2>
              <p class="section-sub">Question generation and answer checking are both driven by your notes.</p>
            </div>
            <div class="controls" style="justify-content:center;margin-bottom:14px;">
              <button class="btn btn-secondary" onclick="prevQuiz()">← Prev</button>
              <button class="btn btn-primary" onclick="generateQuiz()">Generate from Notes</button>
              <button class="btn btn-secondary" onclick="nextQuiz()">Next →</button>
            </div>
            <div id="quizArea"></div>
          </div>
        </div>

        <div class="panel" id="mindmapSection">
          <div class="panel-inner">
            <div class="panel-head">
              <div class="eyebrow">Mind Map</div>
              <h2 class="section-title">See the topic structure visually</h2>
              <p class="section-sub">A concept map is generated from the same saved notes.</p>
            </div>
            <div class="controls" style="margin-bottom:14px;">
              <button class="btn btn-primary" onclick="generateMindmap()">Generate Mind Map</button>
            </div>
            <div id="mindmapArea"></div>
          </div>
        </div>
      </div>
    </section>

    <div class="footer-note">
      Local HTML interface. Notes persist in <span style="font-family:'DM Mono',monospace;">inkwell_notes.json</span>.
    </div>
  </div>

  <script>
    const INITIAL_NOTES = __INITIAL_NOTES__;

    const state = {
      notes: INITIAL_NOTES || [],
      flashcards: [],
      flashcardIndex: 0,
      quiz: [],
      quizIndex: 0,
      quizAnswered: {},
    };

    function escapeHtml(value) {
      return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function setStatus(id, message) {
      document.getElementById(id).textContent = message;
    }

    function renderNotes() {
      const container = document.getElementById('notesList');
      if (!state.notes.length) {
        container.innerHTML = '<div class="empty-state"><div class="empty-emoji">📝</div><div class="empty-title">No notes yet</div><div class="empty-copy">Extract text from an image and save it here.</div></div>';
        return;
      }

      container.innerHTML = state.notes.map((note, index) => {
        const color = ['#c84b2f', '#2f6bc8', '#c8962f', '#2d7a4f', '#8b2fc8'][index % 5];
        const text = String(note.text || '');
        const preview = escapeHtml(text.length > 180 ? text.slice(0, 180) + '…' : text).replace(/\n/g, '<br>');
        return `
          <div class="note-card" style="border-left-color:${color};">
            <div class="note-meta">${escapeHtml(note.date || '')}</div>
            <div class="note-title">${escapeHtml(note.title || 'Untitled')}</div>
            <div class="note-preview">${preview}</div>
            <div class="note-actions"><button class="btn btn-secondary mini" onclick="deleteNote(${index})">Delete</button></div>
          </div>`;
      }).join('');
    }

    function renderFlashcards() {
      const container = document.getElementById('flashcardArea');
      if (!state.flashcards.length) {
        container.innerHTML = '<div class="empty-state"><div class="empty-emoji">🃏</div><div class="empty-title">No flashcards yet</div><div class="empty-copy">Generate from your saved notes.</div></div>';
        return;
      }

      const card = state.flashcards[state.flashcardIndex];
      container.innerHTML = `
        <div class="fc-wrap">
          <div class="fc" onclick="this.classList.toggle('flipped')">
            <div class="fc-face fc-front">
              <div class="fc-label">Question</div>
              <div class="fc-text">${escapeHtml(card.front || '')}</div>
              <div class="fc-hint">Click to flip</div>
            </div>
            <div class="fc-face fc-back">
              <div class="fc-label">Answer</div>
              <div class="fc-text">${escapeHtml(card.back || '')}</div>
            </div>
          </div>
          <div class="fc-counter">${state.flashcardIndex + 1} / ${state.flashcards.length}</div>
        </div>`;
    }

    function prevFlashcard() {
      if (!state.flashcards.length) return;
      state.flashcardIndex = (state.flashcardIndex - 1 + state.flashcards.length) % state.flashcards.length;
      renderFlashcards();
    }

    function nextFlashcard() {
      if (!state.flashcards.length) return;
      state.flashcardIndex = (state.flashcardIndex + 1) % state.flashcards.length;
      renderFlashcards();
    }

    function renderQuiz() {
      const container = document.getElementById('quizArea');
      if (!state.quiz.length) {
        container.innerHTML = '<div class="empty-state"><div class="empty-emoji">❓</div><div class="empty-title">No quiz yet</div><div class="empty-copy">Generate from your saved notes.</div></div>';
        return;
      }

      const q = state.quiz[state.quizIndex];
      const answered = state.quizAnswered[state.quizIndex];
      const dots = state.quiz.map((_, index) => {
        let color = '#d4ccbb';
        if (state.quizAnswered[index] !== undefined) color = '#2d7a4f';
        else if (index === state.quizIndex) color = '#1a1714';
        return `<div style="background:${color};"></div>`;
      }).join('');

      const options = (q.options || []).map((option, index) => {
        let classes = 'option';
        let onclick = '';
        if (answered === undefined) {
          classes += ' neutral';
          onclick = `onclick="answerQuiz(${index})"`;
        } else if (index === q.answer) {
          classes += ' correct';
        } else if (index === answered) {
          classes += ' wrong';
        } else {
          classes += ' neutral';
        }
        return `<button class="${classes}" ${onclick}>${escapeHtml(option)}</button>`;
      }).join('');

      const explanation = answered === undefined || !q.explanation ? '' : `<div class="quiz-explanation">💡 ${escapeHtml(q.explanation)}</div>`;

      container.innerHTML = `
        <div class="quiz-shell">
          <div class="progress-dots">${dots}</div>
          <div class="quiz-card">
            <div class="quiz-question">${escapeHtml(q.question || '')}</div>
            <div class="quiz-options">${options}</div>
            ${explanation}
          </div>
          <div class="quiz-counter">Question ${state.quizIndex + 1} of ${state.quiz.length}</div>
        </div>`;
    }

    function answerQuiz(optionIndex) {
      if (!state.quiz.length) return;
      if (state.quizAnswered[state.quizIndex] !== undefined) return;
      state.quizAnswered[state.quizIndex] = optionIndex;
      renderQuiz();
    }

    function prevQuiz() {
      if (!state.quiz.length) return;
      state.quizIndex = Math.max(0, state.quizIndex - 1);
      renderQuiz();
    }

    function nextQuiz() {
      if (!state.quiz.length) return;
      state.quizIndex = Math.min(state.quiz.length - 1, state.quizIndex + 1);
      renderQuiz();
    }

    async function refreshNotes() {
      const response = await fetch('/api/notes');
      const data = await response.json();
      state.notes = data.notes || [];
      renderNotes();
    }

    async function extractText() {
      const input = document.getElementById('imageInput');
      if (!input.files || !input.files.length) {
        setStatus('extractStatus', 'Upload an image first.');
        return;
      }

      const formData = new FormData();
      formData.append('image', input.files[0]);
      setStatus('extractStatus', 'Running OCR locally and correcting the result...');

      try {
        const response = await fetch('/api/ocr', { method: 'POST', body: formData });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || 'OCR failed');
        }
        document.getElementById('extractedText').value = data.text || '';
        setStatus('extractStatus', data.stats || 'OCR complete.');
        setStatus('saveStatus', 'Text is ready to save.');
      } catch (error) {
        setStatus('extractStatus', 'Error: ' + error.message);
      }
    }

    async function saveCurrentNote() {
      const text = document.getElementById('extractedText').value.trim();
      if (!text) {
        setStatus('saveStatus', 'Nothing to save.');
        return;
      }

      const response = await fetch('/api/notes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const data = await response.json();
      if (!response.ok) {
        setStatus('saveStatus', data.error || 'Unable to save note.');
        return;
      }

      state.notes = data.notes || [];
      renderNotes();
      setStatus('saveStatus', 'Note saved.');
    }

    async function deleteNote(index) {
      const response = await fetch(`/api/notes/${index}`, { method: 'DELETE' });
      const data = await response.json();
      if (!response.ok) {
        alert(data.error || 'Unable to delete note.');
        return;
      }
      state.notes = data.notes || [];
      renderNotes();
    }

    async function clearNotes() {
      const response = await fetch('/api/notes/clear', { method: 'POST' });
      const data = await response.json();
      if (!response.ok) {
        alert(data.error || 'Unable to clear notes.');
        return;
      }
      state.notes = data.notes || [];
      renderNotes();
    }

    async function generateFlashcards() {
      const response = await fetch('/api/flashcards', { method: 'POST' });
      const data = await response.json();
      if (!response.ok) {
        document.getElementById('flashcardArea').innerHTML = `<div class="empty-state"><div class="empty-emoji">⚠️</div><div class="empty-title">Flashcards unavailable</div><div class="empty-copy">${escapeHtml(data.error || 'Generation failed')}</div></div>`;
        return;
      }
      state.flashcards = data.cards || [];
      state.flashcardIndex = 0;
      renderFlashcards();
    }

    async function generateQuiz() {
      const response = await fetch('/api/quiz', { method: 'POST' });
      const data = await response.json();
      if (!response.ok) {
        document.getElementById('quizArea').innerHTML = `<div class="empty-state"><div class="empty-emoji">⚠️</div><div class="empty-title">Quiz unavailable</div><div class="empty-copy">${escapeHtml(data.error || 'Generation failed')}</div></div>`;
        return;
      }
      state.quiz = data.quiz || [];
      state.quizIndex = 0;
      state.quizAnswered = {};
      renderQuiz();
    }

    async function generateMisconceptions() {
      const response = await fetch('/api/misconceptions', { method: 'POST' });
      const data = await response.json();
      if (!response.ok) {
        document.getElementById('misconceptionsArea').innerHTML = `<div class="empty-state"><div class="empty-emoji">⚠️</div><div class="empty-title">Analysis unavailable</div><div class="empty-copy">${escapeHtml(data.error || 'Generation failed')}</div></div>`;
        return;
      }
      document.getElementById('misconceptionsArea').innerHTML = data.html || '';
    }

    async function generateMindmap() {
      const response = await fetch('/api/mindmap', { method: 'POST' });
      const data = await response.json();
      if (!response.ok) {
        document.getElementById('mindmapArea').innerHTML = `<div class="empty-state"><div class="empty-emoji">⚠️</div><div class="empty-title">Mind map unavailable</div><div class="empty-copy">${escapeHtml(data.error || 'Generation failed')}</div></div>`;
        return;
      }
      document.getElementById('mindmapArea').innerHTML = data.html || '';
    }

    renderNotes();
    renderFlashcards();
    renderQuiz();
  </script>
</body>
</html>
"""


@flask_app.get("/")
def index():
    html = HTML_TEMPLATE.replace("__INITIAL_NOTES__", json.dumps(_notes_state, ensure_ascii=False))
    return html


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
    note = {
        "id": len(_notes_state),
        "title": title,
        "text": text,
        "date": datetime.datetime.now().strftime("%b %d, %Y"),
    }
    _notes_state.insert(0, note)
    save_notes(_notes_state)
    return jsonify({"notes": _notes_state, "note": note})


@flask_app.delete("/api/notes/<int:index>")
def api_delete_note(index: int):
    if index < 0 or index >= len(_notes_state):
        return jsonify({"error": "Note index out of range."}), 404

    del _notes_state[index]
    save_notes(_notes_state)
    return jsonify({"notes": _notes_state})


@flask_app.post("/api/notes/clear")
def api_clear_notes():
    _notes_state.clear()
    save_notes(_notes_state)
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


def launch(host="127.0.0.1", port=7860, debug=False, **kwargs):
    """Launch the local HTML app."""
    flask_app.run(host=host, port=port, debug=debug, **kwargs)


if __name__ == "__main__":
    launch()
