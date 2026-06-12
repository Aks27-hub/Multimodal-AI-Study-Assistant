# Multimodal AI Study Assistant

**Handwriting → Notes → Flashcards → Quiz → Mind Maps**

This project is an AI-powered study assistant that converts handwritten notes into structured learning material. It uses **Qwen2.5-VL** for handwriting OCR and **Gemini** for intelligent post-processing and study-tool generation.

---

## Features

### ✍️ Handwriting OCR

* Upload notebook images.
* Uses **Qwen2.5-VL** to transcribe handwritten text.
* Preserves line breaks and document structure.

### 🤖 AI Correction

* Optional Gemini-powered OCR cleanup.
* Fixes:

  * OCR mistakes
  * Character substitutions
  * Misspelled words
  * Formatting issues

### 📝 Notes Management

* Save extracted text as study notes.
* Persistent local storage using JSON.
* Delete individual notes.
* Clear all notes.

### 🃏 Flashcard Generation

Generate AI-powered flashcards from saved notes.

Each flashcard contains:

* Front (question)
* Back (answer)

Useful for active recall and spaced repetition.

### ❓ Quiz Generation

Generate multiple-choice quizzes from notes.

Each question contains:

* Question
* Four answer choices
* Correct answer
* Explanation

### 🔍 Misconception Detection

Gemini analyzes notes and identifies:

* Misconceptions
* Missing concepts
* Ambiguous explanations
* Areas needing clarification

### 🗺️ Mind Map Generation

Automatically creates a visual SVG mind map showing:

* Central topic
* Major concepts
* Subtopics
* Concept relationships

### 🌐 Local Web Interface

Built with Flask and a fully custom HTML/CSS/JavaScript frontend.

No external frontend framework required.

---

# Architecture

```text
Notebook Image
      │
      ▼
Qwen2.5-VL OCR
      │
      ▼
Raw Text
      │
      ▼
Gemini Correction
      │
      ▼
Saved Notes
      │
      ├── Flashcards
      ├── Quiz
      ├── Misconception Analysis
      └── Mind Map
```

---

# Project Structure

```text
.
├── app.py
├── backend.py
├── inkwell_notes.json
├── README.md
└── requirements.txt
```

### app.py

Contains:

* Flask server
* HTML user interface
* API endpoints
* Notes persistence
* Flashcard generation
* Quiz generation
* Mind map rendering
* Misconception analysis

### backend.py

Contains:

* Qwen model loading
* Processor loading
* OCR pipeline
* Gemini client creation
* OCR correction utilities
* Shared AI helper functions

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
pip install google-genai
pip install qwen-vl-utils
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
Qwen/Qwen2.5-VL-3B-Instruct
```

Override:

```bash
export QWEN_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
```

or

```powershell
set QWEN_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
```

---

# Running

Start the application:

```bash
python app.py
```

Default server:

```text
http://127.0.0.1:7860
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

**Gemini 2.5 Flash**

Used for:

* OCR correction
* Flashcards
* Quizzes
* Misconception analysis
* Mind maps

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
