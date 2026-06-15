"""image_analyzer.py

Backend helpers to query LLaVA (llava:7b) via Ollama HTTP API.

Functions:
- detect_image_type
- build_prompt
- build_image_context
- read_questions
- validate_image_and_questions
- ask_llava
- run_image_questions
- run_folder_hardcoded
- dry_run_folder
- get_matching_questions_file

Safe to import from a Streamlit app.
"""
from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None


OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llava:7b"
EXPECTED_PREFIXES = ["drawing_", "text_", "flowchart_"]
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]
NOT_VISIBLE_ANSWER = "Not visible in the image."
UNCERTAINTY_PHRASES = (
    "does not provide",
    "not shown",
    "cannot determine",
    "not enough information",
    "unclear",
    "no information",
    "not specified",
)


def detect_image_type(image_path: str) -> str:
    """Return a simple image type based on filename.

    - If filename contains "drawing" -> "drawing"
    - If filename contains "text" -> "text"
    - If filename contains "flowchart" -> "flowchart"
    - Otherwise -> "general"
    """
    name = Path(image_path).name.lower()
    if "drawing" in name:
        return "drawing"
    if "text" in name:
        return "text"
    if "flowchart" in name:
        return "flowchart"
    return "general"


def build_prompt(question: str, image_type: str, image_context: str = "") -> str:
    """Build a strong instruction prompt for LLaVA.

    The model is instructed to answer ONLY from the image and to reply
    with 'Not visible in the image.' when an answer cannot be found.
    """
    base = (
        "Use the image and the visible context below. If unsure, answer exactly: "
        f"{NOT_VISIBLE_ANSWER} Answer only from visible content. Do not use outside knowledge. "
        "Do not guess. Do not explain. Do not describe the whole image. Give only the direct answer."
    )

    if image_type == "drawing":
        extra = (
            "Focus on visible objects, actions, counts, food, utensils, facial expression, "
            "and spatial relations. Do not infer hidden contents, time, or recipe."
        )
    elif image_type == "text":
        extra = (
            "Read the visible text carefully. Extract exact words, names, titles, relations, "
            "actions, weather, and speech. Do not guess."
        )
    elif image_type == "flowchart":
        extra = (
            "Follow visible arrows and labels exactly. Do not use real-world logic. "
            "If the diagram logic seems strange, still follow the visible arrows."
        )
    else:
        extra = ""

    hint = get_question_hint(question, image_type)

    parts = [base]
    if extra:
        parts.append(extra)
    if image_context:
        parts.append("Visible context:")
        parts.append(image_context)
    if hint:
        parts.append(hint)
    parts.append(f"Question: {question}")
    parts.append("Answer:")
    prompt = "\n".join(parts)
    return prompt


def get_question_hint(question: str, image_type: str) -> str:
    """Return generic extra guidance based on the question wording."""
    lower_question = question.lower()
    hints: List[str] = []

    if "how many" in lower_question or "number of" in lower_question:
        hints.append("Count only clearly visible items. Answer with a number if possible.")
    if any(word in lower_question for word in ("what time", "how long", "threshold", "specific")):
        hints.append("Answer only if an exact visible value is shown. Otherwise answer Not visible in the image.")

    if image_type == "text":
        if any(word in lower_question for word in ("book", "title", "reading")):
            hints.append("Use the exact title from visible context.")
        if any(word in lower_question for word in ("wrote", "author", "by")):
            hints.append("Use the author near 'by'.")
        if any(word in lower_question for word in ("relation", "related", "cousin", "friend", "sister")):
            hints.append("Use explicit relation words only.")
        if "said" in lower_question or "say" in lower_question:
            hints.append("Return the exact spoken sentence by that person.")

    elif image_type == "flowchart":
        if "first step" in lower_question:
            hints.append("Return the first process box after Start, not Start.")
        if "final step" in lower_question or "last step" in lower_question:
            hints.append("Return the last visible process/end step.")
        if "before" in lower_question or "immediately before" in lower_question:
            hints.append("Follow incoming arrows; if multiple, mention multiple paths.")
        if any(word in lower_question for word in ("after", "follows", "next")):
            hints.append("Follow the outgoing arrow from the mentioned step.")
        if "if" in lower_question and "what happens" in lower_question:
            hints.append("Follow the visible Yes/No branch exactly.")

    return " ".join(hints)


def clean_answer(answer: str) -> str:
    """Normalize LLaVA output into the required short answer format."""
    text = (answer or "").strip()
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()

    text = text.replace('"', "'")

    if not text:
        return NOT_VISIBLE_ANSWER

    lower_text = text.lower()
    if any(phrase in lower_text for phrase in UNCERTAINTY_PHRASES):
        return NOT_VISIBLE_ANSWER

    return text[:400]


def read_questions(questions_path: str) -> List[str]:
    """Read non-empty lines from a .txt question file.

    Returns a list of question strings (stripped of surrounding whitespace).
    """
    p = Path(questions_path)
    if not p.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    questions: List[str] = []
    with p.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line:
                questions.append(line)
    return questions


def _image_to_base64(image_path: Path) -> str:
    """Encode image to a plain base64 string for Ollama."""
    data = image_path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def _clean_context(context: str) -> str:
    """Keep image context compact without applying answer-specific rules."""
    text = (context or "").strip()
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace('"', "'")
    return text[:2500]


def _send_ollama_payload(payload: Dict[str, object], timeout: int) -> tuple[Optional[Dict[str, object]], Optional[str]]:
    """Send one Ollama request and return either response data or an error string."""
    if requests is not None:
        try:
            resp = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout)
        except requests.exceptions.ConnectionError:
            return None, "Error: Could not connect to Ollama. Is the Ollama server running on localhost:11434?"
        except requests.exceptions.Timeout:
            return None, "Error: Request to Ollama timed out."
        except Exception as e:
            return None, f"Error: Unexpected request error: {e}"

        if resp.status_code != 200:
            body = resp.text.strip()
            return None, f"Error: Ollama returned status {resp.status_code}: {body}"

        try:
            data = resp.json()
        except Exception:
            return {"response": resp.text.strip()}, None
    else:
        try:
            request_body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                OLLAMA_API_URL,
                data=request_body,
                headers={"Content-Type": "application/json"},
                method="POST",
                )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace").strip()
            return None, f"Error: Ollama returned status {e.code}: {body}"
        except urllib.error.URLError:
            return None, "Error: Could not connect to Ollama. Is the Ollama server running on localhost:11434?"
        except TimeoutError:
            return None, "Error: Request to Ollama timed out."
        except Exception as e:
            return None, f"Error: Unexpected request error: {e}"

        try:
            data = json.loads(body)
        except Exception:
            return {"response": body.strip()}, None

    if isinstance(data, dict):
        return data, None
    return None, "Error: Could not read Ollama response field."


def build_image_context(image_path: str, image_type: str) -> str:
    """Ask LLaVA for visible image context once before answering questions."""
    img_path = Path(image_path)
    if not img_path.exists():
        return ""

    if image_type == "drawing":
        context_prompt = (
            "List only clearly visible facts in this image. Include visible objects, actions, "
            "counts, utensils, food, seasonings, shelf items, counter items, window state, "
            "and facial expression. Do not infer hidden contents, recipe, time, or ingredients. "
            "Use short factual bullet lines."
        )
    elif image_type == "text":
        context_prompt = (
            "Transcribe all visible text exactly as much as possible. Preserve names, titles, "
            "authors, relations, actions, weather, and quoted speech. Do not summarize. "
            "Do not guess missing words."
        )
    elif image_type == "flowchart":
        context_prompt = (
            "Extract the flowchart structure. List all visible nodes and arrows. Include Yes/No "
            "branch labels. Use format: From -> label if any -> To. Follow only visible arrows. "
            "Do not use real-world logic."
        )
    else:
        context_prompt = "List only clearly visible facts. Do not guess."

    payload = {
        "model": MODEL_NAME,
        "prompt": context_prompt,
        "images": [_image_to_base64(img_path)],
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 0.2,
            "num_predict": 300,
        },
    }

    data, error = _send_ollama_payload(payload, timeout=240)
    if error:
        return ""
    if isinstance(data, dict) and isinstance(data.get("response"), str):
        return _clean_context(data["response"])
    return ""


# FIX 2: Switched from .write_text (overwrite) to .open('a') (append) mode.
def _write_results(results: List[Dict[str, str]], output_path: Path) -> None:
    """Write results in the exact required assignment format using Append mode."""
    lines: List[str] = []
    for item in results:
        lines.append(f'picture: "{item["image"]}"')
        lines.append(f'question: "{item["question"]}"')
        lines.append(f'answer: "{item["answer"]}"')
        lines.append("")

    content = "\n".join(lines).rstrip()
    if content:
        content += "\n\n" # Add an extra newline for clean spacing between batches
    
    # Use append mode ('a') instead of write_text to prevent overwriting
    with output_path.open("a", encoding="utf-8") as out_file:
        out_file.write(content)


def _looks_like_title_or_name(answer: str) -> bool:
    """Heuristic for text answers that should be grounded in visible context."""
    if answer in {"", NOT_VISIBLE_ANSWER, "Yes", "No"}:
        return False
    if answer.isdigit():
        return False
    return bool(re.search(r"\b[A-Z][A-Za-z']+(?:\s+[A-Z][A-Za-z']+)*\b", answer))


def _answer_in_context(answer: str, image_context: str) -> bool:
    answer_text = re.sub(r"\s+", " ", answer.lower()).strip(" .,:;!?")
    context_text = re.sub(r"\s+", " ", image_context.lower())
    return bool(answer_text) and answer_text in context_text


def _flowchart_answer_near_context_end(answer: str, image_context: str) -> bool:
    answer_text = re.sub(r"\s+", " ", answer.lower()).strip(" .,:;!?")
    context_text = re.sub(r"\s+", " ", image_context.lower())
    if not answer_text or not context_text:
        return True
    tail_start = max(0, len(context_text) - max(500, len(context_text) // 3))
    return answer_text in context_text[tail_start:]


def _get_retry_instruction(question: str, image_type: str, answer: str, image_context: str = "") -> str:
    """Return one extra instruction for a retry, or an empty string."""
    if answer.startswith("Error:"):
        return ""

    lower_question = question.lower()
    lower_answer = answer.lower().strip()

    if image_type == "flowchart" and "first step" in lower_question and lower_answer == "start":
        return "Do not answer Start. Return the first process box after Start."

    is_final_question = "final step" in lower_question or "last step" in lower_question
    looks_like_beginning = any(word in lower_answer for word in ("start", "begin", "submitted"))
    if image_type == "flowchart" and is_final_question and looks_like_beginning:
        return "Return the last visible step at the end/bottom of the flowchart."

    if image_type == "text" and image_context and _looks_like_title_or_name(answer) and not _answer_in_context(answer, image_context):
        return "Your answer must appear in the visible context. If not, answer Not visible in the image."

    if image_type == "flowchart" and image_context and is_final_question and not _flowchart_answer_near_context_end(answer, image_context):
        return "Return the last visible node reached at the end of the process."

    if len(answer) > 250:
        return "Answer only the direct answer, no explanation."

    return ""


def validate_image_and_questions(image_path: str) -> tuple[bool, str]:
    """Check that the image and its matching questions file are ready."""
    img_path = Path(image_path)
    if not img_path.exists():
        return False, f"Image not found: {image_path}"

    questions_path = img_path.with_suffix(".txt")
    if not questions_path.exists():
        return False, f"Questions file not found: {questions_path}"

    questions = read_questions(str(questions_path))
    if not questions:
        return False, f"Questions file is empty: {questions_path}"

    return True, "OK"


def ask_llava(image_path: str, question: str, timeout: int = 240, image_context: str = "") -> str:
    """Send image+question to Ollama (llava:7b) and return the model's answer string.

    - Uses `OLLAMA_API_URL` and `MODEL_NAME`.
    - Sends the image in the Ollama `images` list as plain base64.
    - Uses `stream: false`.
    - Returns a clean text answer.
    - On errors returns a clear error message.
    - Uses `requests` if it is installed, otherwise uses Python's built-in urllib.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        return f"Error: image not found: {image_path}"

    image_b64 = _image_to_base64(img_path)

    image_type = detect_image_type(image_path)
    if not image_context:
        image_context = build_image_context(image_path, image_type)
    prompt = build_prompt(question, image_type, image_context)

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 0.2,
            "num_predict": 80,
        },
    }

    data, error = _send_ollama_payload(payload, timeout)
    if error:
        return error

    if isinstance(data, dict) and isinstance(data.get("response"), str):
        answer = clean_answer(data["response"])
        retry_instruction = _get_retry_instruction(question, image_type, answer, image_context)
        if retry_instruction:
            retry_payload = dict(payload)
            retry_payload["prompt"] = prompt.replace("\nAnswer:", f"\n{retry_instruction}\nAnswer:")
            retry_data, retry_error = _send_ollama_payload(retry_payload, timeout)
            if retry_error:
                return retry_error
            if isinstance(retry_data, dict) and isinstance(retry_data.get("response"), str):
                return clean_answer(retry_data["response"])
        return answer

    return "Error: Could not read Ollama response field."


def get_matching_questions_file(image_path: str) -> str:
    """Return the matching .txt file path for an image file.

    Replaces the image extension with `.txt`.
    """
    p = Path(image_path)
    return str(p.with_suffix(".txt"))


def _run_questions_without_writing(image_path: Path, questions_path: Path) -> List[Dict[str, str]]:
    """Run all questions for one image and return result dictionaries."""
    questions = read_questions(str(questions_path))
    results: List[Dict[str, str]] = []
    image_type = detect_image_type(str(image_path))
    image_context = build_image_context(str(image_path), image_type)

    for q in questions:
        ans = ask_llava(str(image_path), q, image_context=image_context)
        if isinstance(ans, str) and ans.startswith("Error:"):
            answer_text = ans
        else:
            answer_text = clean_answer(ans)

        results.append({"image": image_path.name, "question": q, "answer": answer_text})

    return results


def run_image_questions(image_path: str, questions_path: str, output_path: Optional[str] = None) -> List[Dict[str, str]]:
    """Run LLaVA on each question for a given image.

    - Reads questions from `questions_path`.
    - Queries LLaVA for each question.
    - Truncates answers to 400 chars.
    - Saves results to `all_answers.txt` in the same folder as the image unless `output_path` is provided.
    - Returns a list of dicts: {"image":..., "question":..., "answer":...}
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if output_path is None:
        out_file = img_path.parent / "all_answers.txt"  # Targets unified underscore naming
    else:
        out_file = Path(output_path)

    results = _run_questions_without_writing(img_path, Path(questions_path))
    _write_results(results, out_file)
    return results


def run_folder_hardcoded(folder_path: str) -> List[Dict[str, str]]:
    """Run drawing_*, text_*, and flowchart_* images found in a folder.

    Looks for png, jpg, and jpeg files. Each image must have a matching .txt
    questions file with the same base name. Writes one `all_answers.txt` file
    in the folder.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder_path}")

    results: List[Dict[str, str]] = []

    for prefix in EXPECTED_PREFIXES:
        matches = [
            p for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
            and p.name.lower().startswith(prefix)
        ]
        matches.sort()

        if not matches:
            continue

        image_path = matches[0]
        ok, message = validate_image_and_questions(str(image_path))
        if not ok:
            raise ValueError(message)

        questions_path = image_path.with_suffix(".txt")
        results.extend(_run_questions_without_writing(image_path, questions_path))

    _write_results(results, folder / "all_answers.txt")  # Targets unified underscore naming
    return results


def dry_run_folder(folder_path: str) -> List[str]:
    """Validate expected folder files without calling LLaVA or writing output."""
    folder = Path(folder_path)
    messages: List[str] = []

    if not folder.exists():
        return [f"Folder not found: {folder_path}"]
    if not folder.is_dir():
        return [f"Not a folder: {folder_path}"]

    for prefix in EXPECTED_PREFIXES:
        matches = [
            p for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
            and p.name.lower().startswith(prefix)
        ]
        matches.sort()

        if not matches:
            messages.append(f"Missing image: {prefix}*.png/jpg/jpeg")
            continue

        image_path = matches[0]
        messages.append(f"Found image: {image_path.name}")

        if len(matches) > 1:
            extra_names = ", ".join(p.name for p in matches[1:])
            messages.append(f"Extra matching images ignored for {prefix}: {extra_names}")

        questions_path = image_path.with_suffix(".txt")
        if not questions_path.exists():
            messages.append(f"Missing questions file: {questions_path.name}")
            continue

        messages.append(f"Found questions file: {questions_path.name}")
        questions = read_questions(str(questions_path))
        if questions:
            messages.append(f"Questions count: {len(questions)}")
        else:
            messages.append(f"Questions file is empty: {questions_path.name}")

    for message in messages:
        print(message)

    return messages


if __name__ == "__main__":
    # Set sample target explicitly to your specific assignment workspace filenames
    sample_folder = Path(".")
    sample_image = "drawing_Robot Chef in Cozy Kitchen.png" 

    img = sample_folder / sample_image
    questions = img.with_suffix(".txt")

    if img.exists() and questions.exists():
        print(f"Running standalone batch verification tests for: {img.name}...")
        res = run_image_questions(str(img), str(questions))
        print(f"Success! Appended data directly into file: {img.parent / 'all_answers.txt'}") 
    else:
        print(f"File configuration check failed. Could not locate: '{sample_image}' inside directory paths.")