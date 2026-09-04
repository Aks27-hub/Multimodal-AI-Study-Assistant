# Multimodal AI Study Assistant

**Handwriting → Notes → RAG Assistant + Flashcards + Quiz + Mind Maps**

This project is an AI-powered study assistant that converts handwritten notes into structured learning material. It uses **Qwen2.5-VL** for local handwriting OCR and **Gemini 3.5 Flash** for intelligent post-processing, RAG-backed Q/A search, and study-tool generation.

---

## Features

### ✍️ Handwriting OCR
* Upload notebook page images.
* Uses local **Qwen2.5-VL** to transcribe handwritten text natively.
* Preserves line breaks and document structure[cite: 3].

### 🤖 AI Correction
* Optional Gemini-powered post-correction pipeline.
* Fixes OCR mistakes, improper noun transcription, character substitutions, and formatting errors[cite: 3].

### 📝 Notes Management
* Save extracted text directly as study notes with auto-generated UUIDs.
* Persistent local storage using JSON (`inkwell_notes.json`)[cite: 1, 2].
* Automatic vector indexing into a local session database upon saving.
* Delete individual notes or clear all session notes[cite: 1, 2].

### 🔍 RAG Query System
* Dual-stage Retrieval-Augmented Generation (RAG) assistant[cite: 1, 2].
* Searches saved session notes via vector store first[cite: 1, 2].
* Falls back to a global knowledge database if local results are insufficient[cite: 1, 2].
* Contextual answers synthesized using Gemini 3.5 Flash.

### 🃏 Flashcard Generation
* Generates flashcards with front/back question pairs directly from saved notes using Gemini[cite: 1, 2].
* Interactive flipping interface with progress tracking[cite: 1].

### ❓ Quiz Generation
* Generates 5-option multiple-choice quizzes with explanations from note context[cite: 1, 2].
* Tracks score, answered questions, and provides visual feedback[cite: 1, 2].

### 💡 Misconception & Gap Analysis
* Analyzes notes for misconceptions, missing concepts, and ambiguous explanations[cite: 1, 2].
* Provides targeted corrections and links back to note source context[cite: 1, 2].

### 🗺️ Visual Mind Map
* Automatically extracts central concepts and subtopics from notes.
* Renders a dynamic, structured SVG mind map[cite: 1, 2].

### 🌐 Flask & HTML Interface
* Served locally via Flask with Jinja template rendering.
* Embedded styling and vanilla JavaScript frontend[cite: 1].

---

# Architecture

```text
Notebook Image
      │
      ▼
Qwen2.5-VL OCR (Local)
      │
      ▼
Gemini Post-Correction
      │
      ▼
Saved Notes ──► ChromaDB Vector Store
      │                │
      ├── Flashcards   └──► RAG Engine ──► Gemini Q/A Synthesis
      ├── Quiz
      ├── Misconceptions
      └── Mind Map

# Project Structure

```text
.
├── app.py              # Flask server routes and API orchestration
├── backend.py          # Qwen2.5-VL OCR & Gemini client pipeline
├── rag_engine.py       # ChromaDB vector store search and retrieval logic
├── templates/
│   └── index.html      # UI template with dynamic JS frontend
├── inkwell_notes.json  # Local JSON persistence for saved notes
└── requirements.txt    # Project dependencies
```
---

# Requirements

## Python

Recommended:

```bash
Python 3.10+
```

---

## Install Dependencies

```bash
pip install flask
pip install torch torchvision accelerate transformers
pip install bitsandbytes
pip install google-genai
pip install qwen-vl-utils
pip install chromadb
```

---

# Environment Variables

## Gemini API Key

Windows:

```powershell
set GEMINI_API_KEY=your_key_here
```

Linux / macOS:

```bash
export GEMINI_API_KEY=your_key_here
```

---

## Optional Qwen Model

Default:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

Override:

```bash
export QWEN_MODEL_PATH=Qwen/Qwen2.5-VL-(2/3)B-Instruct
```

or

```powershell
set QWEN_MODEL_PATH=Qwen/Qwen2.5-VL-(2/3)B-Instruct
```

---

# Running

Start the application:

```bash
python app.py
```

Open the URL in your browser.

---

# API Endpoints

## OCR

```http
POST /api/ocr
```

Upload:

```multipart/form-data
image=<file>
```

Response:

```json
{
  "text": "...",
  "stats": "..."
}
```

---

## Get Notes

```http
GET /api/notes
```

---

## Save Note

```http
POST /api/notes
```

Body:

```json
{
  "text": "my note"
}
```

---

## Delete Note

```http
DELETE /api/notes/<index>
```

---

## Clear Notes

```http
POST /api/notes/clear
```

---

## RAG Generation

```http
POST /api/rag_query
```

---

## Generate Flashcards

```http
POST /api/flashcards
```

---

## Generate Quiz

```http
POST /api/quiz
```

---

## Analyze Misconceptions

```http
POST /api/misconceptions
```

---

## Generate Mind Map

```http
POST /api/mindmap
```

---

# Notes Storage

All notes are stored locally in:

```text
inkwell_notes.json
```

Example:

```json
[
  {
    "id": 0,
    "title": "Introduction to Biology",
    "text": "...",
    "date": "Jun 13, 2026"
  }
]
```

---

# Models Used

## OCR

**Qwen2.5-VL**

* Vision-language model
* Handwriting transcription
* Local inference

## Reasoning & Generation

**Gemini 3.5 Flash**

Used for:

* OCR correction
* Flashcards
* Quizzes
* Misconception analysis
* Mind maps
* RAG Generation

---

# Future Improvements

* PDF upload support
* Spaced repetition scheduling
* User authentication
* Multi-user support
* Note search
* Study analytics dashboard
* Dark mode
* Mobile-responsive UI improvements

---
