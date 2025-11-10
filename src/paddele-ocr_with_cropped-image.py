import os
import cv2
import json
import numpy as np
import tempfile
import pypdfium2 as pdfium
from typing import Dict, List
from PIL import Image
from paddleocr import PaddleOCR
from utils.crop_image import crop_top_and_bottom

# =========================
# PaddleOCR (GPU se disponível)
# =========================
try:
    paddle_ocr = PaddleOCR(
        use_gpu=True,
        lang="pt",  # português
        show_log=False
    )
    print("✅ PaddleOCR inicializado com GPU")
except Exception as e:
    print(f"⚠️ Falha ao inicializar PaddleOCR com GPU: {e}")
    paddle_ocr = PaddleOCR(use_gpu=False, lang="pt", show_log=False)
    print("➡️  Usando CPU (fallback)")


# === Funções auxiliares ===
def _render_pdf_page(pdf, page_index: int, dpi: int = 200):
    """Renderiza uma página do PDF como imagem PIL."""
    page = pdf.get_page(page_index)
    return page.render(scale=dpi / 72).to_pil()


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Converte uma imagem PIL para formato BGR (OpenCV)."""
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def ocr_page_paddle(image_bgr: np.ndarray) -> str:
    """
    Executa OCR via PaddleOCR (versão 3.x compatível).
    Retorna o texto reconhecido concatenado.
    """
    try:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Cria arquivo temporário
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            Image.fromarray(image_rgb).save(tmp.name)

            # Chamada compatível com PaddleOCR >= 3.0
            results = paddle_ocr.ocr(tmp.name)

        if not results or not results[0]:
            return ""

        # Extrai apenas o texto (res[1][0]) para cada box detectado
        lines = [res[1][0].strip() for res in results[0] if res[1][0].strip()]
        return "\n".join(lines)

    except Exception as e:
        raise ValueError(f"Erro no PaddleOCR: {e}")


def process_page(job: Dict, dpi: int = 200) -> Dict:
    """Processa uma página do PDF e extrai o texto via OCR."""
    pdf_path = job["pdf_path"]
    page_index = job["page_index"]

    pdf = pdfium.PdfDocument(pdf_path)
    pil_img = _render_pdf_page(pdf, page_index, dpi)
    bgr_img = _pil_to_bgr(pil_img)

    # Divide em duas metades
    os.makedirs("./data/debug_images", exist_ok=True)
    temp_img_path = f"./data/debug_images/page_{page_index}.png"
    cv2.imwrite(temp_img_path, bgr_img)

    top, bottom = crop_top_and_bottom(temp_img_path)

    # OCR nas duas metades
    text_top = ocr_page_paddle(top)
    text_bottom = ocr_page_paddle(bottom)
    full_text = (text_top + "\n" + text_bottom).strip()

    print(f"\n🧾 Página {page_index + 1} ({len(full_text)} caracteres)")
    print(full_text[:500] + ("..." if len(full_text) > 500 else ""))
    print("-" * 80)

    return {"page": page_index + 1, "text": full_text}


def save_json(data: List[Dict], output_path: str):
    """Salva os resultados em um arquivo JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ Arquivo salvo em: {output_path}")


# === Execução principal ===
if __name__ == "__main__":
    pdf_path = "./data/to_process/08663.pdf"
    output_json_path = "./data/output/08663_paddle-ocr.json"

    pdf = pdfium.PdfDocument(pdf_path)
    n_pages = len(pdf)
    print(f"📘 Total de páginas: {n_pages}")

    results = []
    for i in range(n_pages):
        results.append(process_page({"pdf_path": pdf_path, "page_index": i, "extractor": "paddle-ocr"}))

    save_json(results, output_json_path)
