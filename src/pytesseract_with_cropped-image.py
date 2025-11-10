import os
import cv2
import json
import numpy as np
import pytesseract
import pypdfium2 as pdfium
from typing import Dict, List
from PIL import Image
from utils.crop_image import crop_top_and_bottom

# === Funções auxiliares ===

def _render_pdf_page(pdf, page_index: int, dpi: int = 200):
    """Renderiza uma página do PDF como imagem PIL."""
    page = pdf.get_page(page_index)
    pil_image = page.render(scale=dpi / 72).to_pil()
    return pil_image


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Converte uma imagem PIL para formato BGR (OpenCV)."""
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def tesseract_ocr(img: np.ndarray) -> str:
    """Executa OCR com Tesseract em uma imagem numpy."""
    try:
        cfg = "--oem 1 --psm 3"
        text = pytesseract.image_to_string(img, lang="por", config=cfg)
        return text.strip()
    except Exception as e:
        raise ValueError(f"Erro ao ler o crop da imagem: {e}")


def process_page(job: Dict, dpi: int = 200) -> Dict:
    """
    - Renderiza a página do PDF.
    - Converte para imagem OpenCV.
    - Corta metade superior e inferior.
    - Executa OCR em cada parte.
    - Retorna dicionário com o texto da página.
    """
    pdf_path = job["pdf_path"]
    page_index = job["page_index"]

    # Reabre o PDF (cada processo precisa de sua própria instância)
    pdf = pdfium.PdfDocument(pdf_path)
    pil_img = _render_pdf_page(pdf, page_index, dpi)
    bgr_img = _pil_to_bgr(pil_img)

    # Caminho temporário
    os.makedirs("./data/debug_images", exist_ok=True)
    temp_img_path = f"./data/debug_images/page_{page_index}.png"
    cv2.imwrite(temp_img_path, bgr_img)

    # Divide em duas metades
    top, bottom = crop_top_and_bottom(temp_img_path)

    # OCR nas duas partes
    text_top = tesseract_ocr(top)
    text_bottom = tesseract_ocr(bottom)
    full_text = text_top + "\n" + text_bottom

    print(f"\n🧾 Página {page_index + 1}:")
    print(full_text)
    print("-" * 80)

    return {"page": page_index + 1, "text": full_text}


def save_json(data: List[Dict], output_path: str):
    """Salva a lista de dicionários em JSON formatado."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ Arquivo salvo em: {output_path}")


# === Execução principal ===

if __name__ == "__main__":
    pdf_path = "./data/to_process/08663.pdf"
    output_json_path = "./data/output/08663.json"

    pdf = pdfium.PdfDocument(pdf_path)
    n_pages = len(pdf)

    print(f"📘 Total de páginas: {n_pages}")

    results = []
    for page_index in range(n_pages):
        result = process_page({"pdf_path": pdf_path, "page_index": page_index})
        results.append(result)

    save_json(results, output_json_path)
