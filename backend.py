"""
Local setup:
    - Set QWEN_MODEL_PATH to a Hugging Face model id or a local model directory.
    - Set GEMINI_API_KEY if you want the optional post-correction step.

Example:
    QWEN_MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
    GEMINI_API_KEY="your-key"
"""

import os
import torch
import google.genai as genai
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# ── Config ────────────────────────────────────────────────────────────────────
# Set the GEMINI MODEL you want
GEMINI_MODEL   = "gemini-2.5-flash"

# Model options (pick one):
#   "Qwen/Qwen2.5-VL-7B-Instruct"   — best accuracy, needs ~16GB VRAM
#   "Qwen/Qwen2.5-VL-3B-Instruct"   — good accuracy, needs ~8GB VRAM
#   "Qwen/Qwen2.5-VL-2B-Instruct"   — lighter, runs on CPU (slow but works)

model_path = os.getenv("QWEN_MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")

def get_qwen():
    if torch.cuda.is_available():
        # Using 4-bit quantization to fit larger models in less VRAM (with some accuracy tradeoff).
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    else:
        quantization_config = None

    kwargs = {"device_map": "auto"}
    if quantization_config:
        kwargs["quantization_config"] = quantization_config
    else:
        kwargs["torch_dtype"] = torch.float32
    
    print(f"Loading {model_path}...")
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        **kwargs
    )
    return qwen_model

def get_processor():
    qwen_processor = AutoProcessor.from_pretrained(model_path)
    return qwen_processor

qwen_model = get_qwen()
qwen_processor = get_processor()

def get_gemini_client():
    sec_val = os.getenv("GEMINI_API_KEY")
    if not sec_val:
        print("[Warning] GEMINI_API_KEY not set, skipping LLM correction.")
        return None
    client = genai.Client(api_key=sec_val)
    return client

client = get_gemini_client()

# ── Step 1: Qwen2.5-VL transcription:
def transcribe_with_qwen(qwen_model, qwen_processor, image_path: str) -> list[str]:
    """
    Feed the whole image to Qwen2.5-VL and get back transcribed lines.
    No line segmentation, no preprocessing — Qwen handles it natively.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{image_path}",
                },
                {
                    "type": "text",
                    "text": (
                        "This is a handwritten page from an Indian school notebook. "
                        "Transcribe ALL the handwritten text exactly as written, "
                        "preserving the original line breaks. "
                        "Output only the transcribed text, nothing else."
                    ),
                },
            ],
        }
    ]

    # Build inputs
    text_prompt = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = qwen_processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(qwen_model.device)

    # Generate
    with torch.no_grad():
        output_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,       # greedy — most consistent for OCR
        )

    # Decode (strip the input prompt tokens, keep only generated part)
    generated_ids = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, output_ids)
    ]
    
    raw_output = qwen_processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    # Split into lines, strip empty lines
    lines = [ln.strip() for ln in raw_output.strip().splitlines() if ln.strip()]
    return lines


# ── Step 2: LLM correction (1 API call) ──────────────────────
def correct_with_llm(raw_lines: list[str]) -> list[str]:
    """
    Post-correct Qwen output with Gemini.
    mainly catches proper nouns and rare character confusions.
    """
    if not client:
        return raw_lines

    numbered_text = "\n".join(f"{i+1}. {line}" for i, line in enumerate(raw_lines))

    prompt = f"""You are correcting OCR output from a handwritten Indian school notebook page.
The OCR model (Qwen2.5-VL) is generally accurate but may make occasional errors.

Common error types to fix:
- Misread proper nouns (Indian names, place names)
- Occasional character swaps (0/O, 1/l/I, rn/m)
- Missing or extra punctuation
- Words run together or split incorrectly

TASK:
- Fix any OCR errors in the numbered lines below
- Preserve original meaning, structure, line breaks, and numbers exactly
- You MUST return ALL {len(raw_lines)} lines, even if a line needs no correction
- If a line looks correct, copy it exactly as-is
- Never skip, merge, or summarize lines
- Output ONLY the corrected numbered lines, nothing else

OCR OUTPUT:
{numbered_text}"""

    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    except Exception as e:
        print(e)
        return raw_lines

    if not response.text:
        return raw_lines
    corrected_raw = response.text.strip()

    # Parse numbered lines back to a list
    line_dict = {}
    for line in corrected_raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit() and ". " in line:
            num, text = line.split(". ", 1)
            try:
                # Store in dict to handle potential line order changes, but we will re-order later
                line_dict[int(num)] = text
            except ValueError:
                pass

    # Safety fallback: if line count changed, pad/trim
    corrected_lines = []
    for i in range(1, len(raw_lines) + 1):
        if i in line_dict:
            corrected_lines.append(line_dict[i])
        else:
            # LLM skipped this line — keep raw and log it
            print(f"  [Warning] LLM skipped line {i}, keeping raw: {raw_lines[i-1]}")
            corrected_lines.append(raw_lines[i - 1])

    return corrected_lines


# ── Step 3: Full pipeline ─────────────────────────────────────────────────────
def recognize(image_path: str, use_llm: bool = True) -> str:
    """
    Full pipeline: Qwen2.5-VL transcription → Gemini correction.

    API calls made per execution:
      - Qwen2.5-VL: runs locally (0 API calls)
      - Gemini correction: 1 API call (entire document in one request)
      Total: 1 API call
    """
    print(f"Processing: {image_path}\n")
    
    print("Model loaded.\n")

    # --- Qwen transcription (local, no API) ---
    print("Step 1: Qwen2.5-VL transcription...")
    raw_lines = transcribe_with_qwen(qwen_model, qwen_processor, image_path)

    print(f"  Detected {len(raw_lines)} lines\n")
    for i, line in enumerate(raw_lines):
        print(f"  Line {i+1:2d} [raw]:  {line}")

    if not use_llm:
        result = "\n".join(raw_lines)
        print("\n=== Final Output (no LLM correction) ===")
        print(result)
        return result

    # --- Gemini correction (1 API call) ---
    print("\nStep 2: Gemini correction (1 API call)...")
    corrected_lines = correct_with_llm(raw_lines)

    print("\n--- Diff (raw → corrected) ---")
    for i, (raw, cor) in enumerate(zip(raw_lines, corrected_lines)):
        tag = "✓ fixed" if raw != cor else "  same "
        print(f"  Line {i+1:2d} [{tag}]  {cor}")

    final = "\n".join(corrected_lines)
    print("\n=== Final Output ===")
    print(final)
    return final


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Change to your image path:
    image_path = "your_image.jpg"
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    recognize(image_path)
