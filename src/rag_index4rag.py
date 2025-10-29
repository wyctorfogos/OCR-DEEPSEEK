import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# =========================
# CONFIG
# =========================
OUTPUT_DIR = Path("./data/output")  # onde estão seus .json do OCR
EMBED_MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
# -> troque pra um modelo que você já tenha local em PT-BR, ex:
# "sentence-transformers/multi-qa-mpnet-base-dot-v1"
# ou um modelo seu fine-tunado.


# =========================
# Helpers RAG
# =========================

def load_pages_from_json(json_path: Path) -> List[Dict]:
    """
    Lê um arquivo JSON no formato:
    [
      {"page": 1, "text": "..."},
      {"page": 2, "text": "..."}
    ]
    Retorna essa lista (páginas) com metadados anexados.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # anexa nome do arquivo original como 'source_pdf'
    for item in data:
        item["source_pdf"] = json_path.stem  # ex: "documento123"
    return data


def chunk_text(text: str,
               max_chars: int = 1000,
               overlap: int = 200) -> List[str]:
    """
    Quebra um texto longo em janelas de até max_chars caracteres,
    com 'overlap' para não cortar contexto bruscamente.
    """
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # volta um pouco pra manter contexto
        if start < 0:
            start = 0
        if start >= length:
            break
    return chunks


def build_corpus(output_dir: Path) -> List[Dict]:
    """
    Percorre todos os .json em data/output e constrói uma lista de chunks.
    Cada chunk vira um "documento recuperável" no RAG.
    Estrutura de cada item:
    {
        "text": "...",
        "source_pdf": "arquivo",
        "page": 3
    }
    """
    corpus = []
    for filename in os.listdir(output_dir):
        if not filename.endswith(".json"):
            continue

        json_path = output_dir / filename
        pages = load_pages_from_json(json_path)

        for p in pages:
            page_num = p["page"]          # 1-based
            page_txt = p["text"] or ""
            source_pdf = p["source_pdf"]  # base sem extensão

            # quebra a página em subtrechos
            for chunk in chunk_text(page_txt, max_chars=1000, overlap=200):
                corpus.append({
                    "text": chunk,
                    "source_pdf": source_pdf,
                    "page": page_num
                })

    return corpus


def encode_corpus(corpus: List[Dict],
                  model: SentenceTransformer) -> np.ndarray:
    """
    Gera embeddings para cada chunk de texto do corpus.
    Retorna array (N, D).
    """
    texts = [doc["text"] for doc in corpus]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Cria um índice FAISS de similaridade por produto interno (cosine approx).
    Vamos normalizar os vetores antes pra simular cosine similarity.
    """
    # normaliza embeddings para norma 1 (na verdade norma L2 = 1)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    normed = embeddings / norms

    dim = normed.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product
    index.add(normed)

    return index, normed


def search(query: str,
           model: SentenceTransformer,
           index: faiss.IndexFlatIP,
           normed_embeddings: np.ndarray,
           corpus: List[Dict],
           top_k: int = 3) -> List[Dict]:
    """
    Faz busca semântica: retorna os top_k chunks mais parecidos com a query.
    """
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)

    sims, ids = index.search(q_emb, top_k)
    sims = sims[0]
    ids = ids[0]

    results = []
    for score, idx in zip(sims, ids):
        doc = corpus[idx]
        results.append({
            "score": float(score),
            "text": doc["text"],
            "source_pdf": doc["source_pdf"],
            "page": doc["page"]
        })
    return results


def build_prompt(query: str, retrieved_chunks: List[Dict]) -> str:
    """
    Monta um prompt estilo RAG pra mandar pra um LLM.
    Você passa esse prompt pro seu modelo (Gemma, Qwen local, etc.).
    """
    context_blocks = []
    for r in retrieved_chunks:
        block = (
            f"[Fonte: {r['source_pdf']} pág {r['page']}]\n"
            f"{r['text']}\n"
        )
        context_blocks.append(block)

    context_text = "\n---\n".join(context_blocks)

    prompt = f"""
Você é um assistente que responde SOMENTE com base no contexto fornecido.
Se não tiver informação suficiente no contexto, diga que não sabe.

Pergunta do usuário:
{query}

Contexto relevante:
{context_text}

Resposta:
"""
    return prompt.strip()


if __name__ == "__main__":
    # 1. monta o corpus a partir dos JSONs OCR
    corpus = build_corpus(OUTPUT_DIR)

    # 2. carrega o modelo de embedding
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # 3. gera embeddings e índice FAISS
    embeddings = encode_corpus(corpus, model)
    index, normed_embeddings = build_faiss_index(embeddings)

    # 4. faz uma pergunta de teste
    user_query = "Qual é o número do processo e a vara?"
    retrieved = search(
        query=user_query,
        model=model,
        index=index,
        normed_embeddings=normed_embeddings,
        corpus=corpus,
        top_k=3
    )

    # 5. gera o prompt final para LLM
    prompt_for_llm = build_prompt(user_query, retrieved)

    print("========== PROMPT PARA O LLM ==========")
    print(prompt_for_llm)
    print("=======================================")
