# vlm_rag_extract.py
import os
import json
import glob
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, ValidationError
from ollama import chat

# =========================
# CONFIG
# =========================
OCR_JSON_DIR = Path("data/output")        # where your JSON OCR lives
PAGE_IMG_DIR = Path("data/page_images")   # optional: page images like 06520_page_1.png
EMBED_MODEL  = "neuralmind/bert-base-portuguese-cased"
VLM_MODEL    = os.getenv("VLM_MODEL", "gemma3:4b")  # or 'qwen2.5-vl:7b-instruct'
TOP_K        = 5
CHUNK_SIZE   = 600
CHUNK_OVERLAP = 120

# =========================
# UTILS
# =========================
def load_ocr_json(json_path: Path) -> List[Dict[str, Any]]:
    """
    Your sample format:
    [
      {"page": 1, "text": "Dados a serem projetados"},
      {"page": 2, "text": "Texto extraído"}
    ]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def build_corpus(ocr_json_dir: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Returns:
      texts: list of chunk strings
      metadatas: parallel list with {source, page, chunk_id}
    """
    texts, metas = [], []
    for jpath in sorted(ocr_json_dir.glob("*.json")):
        doc_id = jpath.stem  # e.g., "06520"
        pages = load_ocr_json(jpath)
        for p in pages:
            page_no = p.get("page")
            page_text = p.get("text", "")
            for i, chunk in enumerate(chunk_text(page_text)):
                texts.append(chunk)
                metas.append({"source": doc_id, "page": int(page_no), "chunk_id": i})
    return texts, metas

def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    emb = SentenceTransformer(model_name)
    embs = emb.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype(np.float32)

def faiss_index(embs: np.ndarray) -> faiss.Index:
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine with normalized vectors -> inner product
    index.add(embs)
    return index

def find_page_images(source_id: str, page_numbers: List[int]) -> List[str]:
    """
    Looks for images like: data/page_images/{source_id}_page_{page}.png
    Returns existing files only. If none found, returns empty list (text-only mode).
    """
    images = []
    for p in sorted(set(page_numbers)):
        candidate = PAGE_IMG_DIR / f"{source_id}_page_{p}.png"
        if candidate.exists():
            images.append(str(candidate))
    return images

def retrieve(query: str, emb_model: SentenceTransformer, index: faiss.Index, texts: List[str], metas: List[Dict[str, Any]], k: int = TOP_K):
    q_emb = emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    D, I = index.search(q_emb, k)
    I = I[0].tolist()
    hits = [(texts[i], metas[i], float(D[0][pos])) for pos, i in enumerate(I) if i >= 0]
    return hits

# =========================
# SCHEMA EXAMPLE (customize!)
# =========================
class FormSchema(BaseModel):
    company_name: Optional[str] = None
    cnpj: Optional[str] = None
    cpf: Optional[str] = None
    address: Optional[str] = None
    signature_date: Optional[str] = None

def build_schema_prompt(schema_cls: BaseModel) -> str:
    fields = schema_cls.model_fields.keys()  # pydantic v2
    required_list = [f for f, spec in schema_cls.model_fields.items() if spec.is_required()]
    return (
        "Return ONLY a valid JSON object matching this schema.\n"
        "If a field is not present, return it as null.\n"
        "Do NOT include any text outside the JSON.\n\n"
        f"Fields: {list(fields)}\n"
        f"Required fields: {required_list if required_list else 'none'}\n"
        "Example output shape (values illustrative):\n"
        "{\n"
        '  "company_name": "ACME Ltda",\n'
        '  "cnpj": "12.345.678/0001-99",\n'
        '  "cpf": "123.456.789-00",\n'
        '  "address": "Rua Exemplo, 123 - Vitória/ES",\n'
        '  "signature_date": "2024-07-12"\n'
        "}\n"
    )

# =========================
# VLM CALL
# =========================
def ask_vlm_with_rag(
    query: str,
    retrieved: List[Tuple[str, Dict[str, Any], float]],
    schema_prompt: str,
    vlm_model: str = VLM_MODEL,
    include_images: bool = True
) -> str:
    """
    Builds a single VLM prompt that:
      - Shows the user query
      - Adds top-k retrieved text chunks as context (with page/source)
      - Optionally attaches page images corresponding to those chunks
      - Asks for strict JSON according to `schema_prompt`
    Returns the raw model string (expected to be JSON).
    """
    # Consolidate a small context
    lines = []
    pages_for_images: Dict[str, List[int]] = {}
    for text, meta, score in retrieved:
        doc = meta["source"]
        page = meta["page"]
        lines.append(
            f"[source={doc} page={page} score={score:.3f}]\n{text}\n"
        )
        pages_for_images.setdefault(doc, []).append(page)

    system_msg = (
        "You are a precise document information extraction assistant. "
        "Use ONLY the provided OCR context and (if present) the attached page images. "
        "Do not assume facts that are not visible in the OCR or the image."
    )

    user_msg = (
        f"User query:\n{query}\n\n"
        "OCR context (top-k retrieved):\n"
        + "\n---\n".join(lines)
        + "\n\n"
        + schema_prompt
    )

    # Attach images (if available)
    images_to_send: List[str] = []
    if include_images:
        for doc, pages in pages_for_images.items():
            imgs = find_page_images(doc, pages)
            images_to_send.extend(imgs)

    # Build messages (Ollama multimodal expects images in user content)
    messages = [{"role": "system", "content": system_msg}]
    if images_to_send:
        # Some VLMs want images in the same user message, others allow multiple.
        messages.append({"role": "user", "content": user_msg, "images": images_to_send})
    else:
        messages.append({"role": "user", "content": user_msg})

    resp = chat(model=vlm_model, messages=messages, options={"temperature": 0.0})
    return resp["message"]["content"]

def safe_parse_json(txt: str) -> Optional[Dict[str, Any]]:
    """
    Try to coerce to JSON object. Simple guard against stray text.
    """
    txt = txt.strip()
    # heuristic: cut before/after the first {...} block if model added extra text
    if txt.count("{") >= 1 and txt.count("}") >= 1:
        start = txt.find("{")
        end   = txt.rfind("}")
        candidate = txt[start:end+1]
    else:
        candidate = txt
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None

# =========================
# MAIN (index + query)
# =========================
def main():
    # 1) Build corpus
    print("🔎 Loading OCR JSONs...")
    texts, metas = build_corpus(OCR_JSON_DIR)
    if not texts:
        print(f"[WARN] No OCR text found in {OCR_JSON_DIR}.")
        return

    # 2) Embeddings + FAISS
    print("🧮 Embedding texts...")
    emb_model = SentenceTransformer(EMBED_MODEL)
    embs = emb_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    print("📚 Building FAISS index...")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    # 3) Example query + schema
    query = "Extraia company_name, CNPJ/CPF, endereço e data de assinatura."
    print(f"❓ Query: {query}")

    hits = retrieve(query, emb_model, index, texts, metas, k=TOP_K)
    if not hits:
        print("[WARN] No retrieval hits.")
        return

    # 4) Schema prompt (customize fields in FormSchema)
    schema_prompt = build_schema_prompt(FormSchema)

    # 5) Ask VLM with RAG
    print("🧠 Asking VLM (with RAG context)...")
    raw = ask_vlm_with_rag(query, hits, schema_prompt, vlm_model=VLM_MODEL, include_images=True)
    print("---- RAW MODEL OUTPUT ----")
    print(raw)
    print("--------------------------")

    # 6) Parse + validate
    data = safe_parse_json(raw)
    if data is None:
        print("[ERROR] Could not parse JSON from model output.")
        return

    try:
        obj = FormSchema(**data)
    except ValidationError as e:
        print("[ERROR] JSON does not match schema:", e)
        print("Model JSON:\n", json.dumps(data, ensure_ascii=False, indent=2))
        return

    # 7) Done
    print("✅ Parsed & validated:")
    print(json.dumps(obj.model_dump(), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
