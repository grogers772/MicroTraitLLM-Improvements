import os
import uuid
import json
import base64
from typing import Dict, Any

from openai import OpenAI
from read_api_keys import load_api_keys

# Storage locations
DATA_DIR = "data"
FIGURE_DIR = os.path.join(DATA_DIR, "figures")
FIGURE_INDEX_PATH = os.path.join(DATA_DIR, "figure_index.jsonl")

# Make sure directories exist
os.makedirs(FIGURE_DIR, exist_ok=True)

# OpenAI client for vision + embeddings
_api_keys = load_api_keys("apikeys.txt")
_api_key_openai = _api_keys.get("API_KEY_OPENAI")
client = OpenAI(api_key=_api_key_openai)


def _extract_metadata_from_image(image_bytes: bytes) -> Dict[str, Any]:
    """
    Call a multimodal model to extract:
      - caption: concise description of the figure
      - ocr_text: any text / labels it can read
      - chart_data: table-like structure of key numbers, if present
    Returns a dict.
    """
    if not _api_key_openai:
        raise RuntimeError("API_KEY_OPENAI missing in apikeys.txt")

    encoded = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{encoded}"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a vision model analyzing scientific figures. "
                    "Return a JSON object with fields: "
                    "`caption` (string), "
                    "`ocr_text` (string of all readable text), and "
                    "`chart_data` (an array of rows; each row an object with keys like "
                    "axis_labels, series, values, units, etc.). "
                    "If something is missing, use an empty string or empty array."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this figure and extract caption, OCR text, and structured chart/table data as JSON.",
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0.0,
    )

    content = resp.choices[0].message.content
    try:
        metadata = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: wrap raw content
        metadata = {
            "caption": content,
            "ocr_text": "",
            "chart_data": [],
        }

    # Ensure keys exist
    metadata.setdefault("caption", "")
    metadata.setdefault("ocr_text", "")
    metadata.setdefault("chart_data", [])

    return metadata


def _embed_text(text: str) -> list:
    """
    Get a text embedding for caption + OCR text.
    Used for the text index (and as a stand-in for an image embedding for now).
    """
    if not _api_key_openai:
        raise RuntimeError("API_KEY_OPENAI missing in apikeys.txt")

    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
    )
    return resp.data[0].embedding


def ingest_figure(image_bytes: bytes, paper_id: str | None = None) -> Dict[str, Any]:
    """
    Main ingestion entry point.

    1) Run vision model to get metadata (caption, ocr_text, chart_data).
    2) Compute embeddings.
    3) Save original image to disk.
    4) Append a record to figure_index.jsonl.

    Returns the stored record, including a generated figure_id.
    """
    # Step 1: metadata from vision model
    metadata = _extract_metadata_from_image(image_bytes)

    # Step 2: embeddings
    text_for_embedding = (metadata.get("caption", "") + "\n" +
                          metadata.get("ocr_text", "")).strip()
    text_embedding = _embed_text(text_for_embedding) if text_for_embedding else []

    # For now, we'll just reuse text_embedding for image_index as well.
    image_embedding = text_embedding

    # Step 3: save original image
    figure_id = str(uuid.uuid4())
    image_filename = f"{figure_id}.png"
    image_path = os.path.join(FIGURE_DIR, image_filename)
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    # Step 4: build record
    record: Dict[str, Any] = {
        "figure_id": figure_id,
        "paper_id": paper_id,
        "caption": metadata.get("caption", ""),
        "ocr_text": metadata.get("ocr_text", ""),
        "chart_data": metadata.get("chart_data", []),
        "text_embedding": text_embedding,
        "image_embedding": image_embedding,
        "image_path": image_path,
    }

    # Append to JSONL index
    with open(FIGURE_INDEX_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record
