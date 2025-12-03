import base64
import json
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI, RateLimitError
from groq import Groq
from read_api_keys import load_api_keys

# -------------------------------------------------
# Load API keys and create clients
# -------------------------------------------------

api_keys = load_api_keys("apikeys.txt")
api_key_openai = api_keys.get("API_KEY_OPENAI")
api_key_groq = api_keys.get("API_KEY_GROQ")

client_openai: Optional[OpenAI] = OpenAI(api_key=api_key_openai) if api_key_openai else None
client_groq: Optional[Groq] = Groq(api_key=api_key_groq) if api_key_groq else None


# -------------------------------------------------
# Figure metadata "ingestion" stub
# -------------------------------------------------

@dataclass
class FigureMetadata:
    caption: str
    ocr_text: str
    chart_data: str


def ingest_figure(data_url: str) -> Optional[FigureMetadata]:
    """
    Lightweight ingestion step:
    - Uses OpenAI vision to extract a caption, OCR-style text, and chart/table info.
    - Returns a FigureMetadata object, or None if ingestion fails.
    """
    if not client_openai:
        # No OpenAI key -> skip ingestion
        return None

    try:
        resp = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a tool that extracts structured metadata from scientific figures. "
                        "Return ONLY valid JSON with keys 'caption', 'ocr_text', and 'chart_data'. "
                        "Keep each field concise but informative."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Look at this figure and extract:\n"
                                "1) A short caption summarizing what it shows.\n"
                                "2) OCR-like text: any labels, axis titles, or visible text.\n"
                                "3) Chart/table data: a brief description of trends, ranges, or key values.\n\n"
                                "Respond ONLY as JSON."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            temperature=0.0,
        )

        raw = resp.choices[0].message.content or ""

        try:
            meta = json.loads(raw)
            return FigureMetadata(
                caption=(meta.get("caption") or "").strip(),
                ocr_text=(meta.get("ocr_text") or "").strip(),
                chart_data=(meta.get("chart_data") or "").strip(),
            )
        except json.JSONDecodeError:
            # If the model doesn't obey JSON instructions, treat entire content as a caption.
            return FigureMetadata(
                caption=raw.strip(),
                ocr_text="",
                chart_data="",
            )

    except RateLimitError:
        # If OpenAI is rate-limited, just skip ingestion instead of crashing.
        return None
    except Exception:
        # Any other ingestion error -> return None (VQA can still run on raw image)
        return None


# -------------------------------------------------
# Main VQA function with OpenAI -> Groq fallback
# -------------------------------------------------

def run_vqa(image_file, question: str) -> str:
    """
    High-level VQA helper.

    Parameters
    ----------
    image_file : werkzeug.datastructures.FileStorage
        The uploaded image from Flask (request.files["image"])
    question : str
        User's question about the figure

    Returns
    -------
    str
        Either an answer string or a readable error message.
    """
    if not question:
        return "No question provided."

    # Read raw bytes from the uploaded image
    image_bytes = image_file.read()
    if not image_bytes:
        return "Uploaded image is empty or could not be read."

    # Encode to base64 so we can send it as a data URL (for OpenAI vision)
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{encoded}"

    # 1) Ingestion step: try to extract metadata from the figure (caption, OCR, chart data)
    metadata = ingest_figure(data_url)

    meta_text_block = ""
    if metadata:
        bits = []
        if metadata.caption:
            bits.append(f"Caption: {metadata.caption}")
        if metadata.ocr_text:
            bits.append(f"OCR-like text: {metadata.ocr_text}")
        if metadata.chart_data:
            bits.append(f"Chart / table data: {metadata.chart_data}")
        if bits:
            meta_text_block = "Here is pre-extracted metadata about the figure:\n" + "\n".join(bits)

    openai_error_msg = None
    groq_error_msg = None

    # 2) Primary path: OpenAI multimodal VQA (uses raw image + metadata)
    if client_openai:
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that explains scientific figures, "
                        "especially in microbiology / microbial traits. "
                        "Use both the raw image and any provided metadata. "
                        "Be concise but clear, and focus on what the figure shows."
                    ),
                },
            ]

            user_content = []
            if meta_text_block:
                user_content.append({"type": "text", "text": meta_text_block + "\n\n"})
            user_content.append({"type": "text", "text": question})
            user_content.append({"type": "image_url", "image_url": {"url": data_url}})

            messages.append({"role": "user", "content": user_content})

            resp = client_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0,
            )

            answer = resp.choices[0].message.content
            if answer:
                return answer

            openai_error_msg = "Empty response from OpenAI VQA."

        except RateLimitError as e:
            openai_error_msg = f"OpenAI rate/credit limit error: {e}"
        except Exception as e:
            openai_error_msg = f"OpenAI VQA error: {e}"

    # 3) Fallback: Groq text-only VQA using the metadata (no direct image access)
    #
    # This still demonstrates the 'image-aware path' from the architecture:
    # image -> ingestion -> metadata -> answerer
    if client_groq and metadata:
        try:
            groq_prompt_parts = [
                "You answer questions about a scientific figure.",
                "You DO NOT see the original image, only pre-extracted metadata.",
                "",
                "Figure metadata:",
                f"- Caption: {metadata.caption or '(none)'}",
                f"- OCR-like text: {metadata.ocr_text or '(none)'}",
                f"- Chart / table data: {metadata.chart_data or '(none)'}",
                "",
                f"Question: {question}",
                "",
                "Answer based only on this metadata. If the metadata is insufficient, say so explicitly."
            ]

            groq_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a careful model that answers questions about scientific figures "
                        "using only structured metadata (caption, OCR text, extracted chart data). "
                        "Do NOT hallucinate details that are not supported by the metadata."
                    ),
                },
                {"role": "user", "content": "\n".join(groq_prompt_parts)},
            ]

            resp_groq = client_groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=groq_messages,
                temperature=0.0,
            )

            g_answer = resp_groq.choices[0].message.content
            if g_answer:
                return (
                    g_answer
                    + "\n\n[Note: This fallback answer was generated from extracted metadata only; "
                    + "the Groq model did not see raw image pixels.]"
                )

            groq_error_msg = "Empty response from Groq (metadata-only) VQA."

        except Exception as e:
            groq_error_msg = f"Groq VQA error: {e}"

    # 4) If everything fails, return a combined error string
    error_lines = ["VQA failed with all providers."]
    if openai_error_msg:
        error_lines.append(openai_error_msg)
    if groq_error_msg:
        error_lines.append(groq_error_msg)

    return "\n".join(error_lines)
