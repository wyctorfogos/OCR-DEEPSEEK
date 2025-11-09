import os
import cv2
import numpy as np
import pytesseract
import pypdfium2 as pdfium
from typing import Dict
from utils.crop_image import crop_top_and_bottom
from PIL import Image

def _render_pdf_page(pdf, page_index: int, dpi: int = 200):
    """Renderiza uma página do PDF como imagem PIL."""
    page = pdf.get_page(page_index)
    pil_image = page.render(scale=dpi / 72).to_pil()
    return pil_image

def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Converte uma imagem PIL para formato BGR (para OpenCV)."""
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def tesseract_ocr(img: np.ndarray) -> str:
    """Executa OCR com Tesseract em uma imagem numpy."""
    try:
        cfg = "--oem 3 --psm 3"
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
    """
    pdf_path = job["pdf_path"]
    page_index = job["page_index"]

    # Reabre o PDF (cada processo precisa ter sua própria instância)
    pdf = pdfium.PdfDocument(pdf_path)
    pil_img = _render_pdf_page(pdf, page_index, dpi)
    bgr_img = _pil_to_bgr(pil_img)

    # Salva temporariamente a imagem para reutilizar a função existente
    temp_img_path = f"./data/debug_images/page_{page_index}.png"
    os.makedirs(os.path.dirname(temp_img_path), exist_ok=True)
    cv2.imwrite(temp_img_path, bgr_img)

    # Divide a imagem em duas metades
    top, bottom = crop_top_and_bottom(temp_img_path)

    # OCR em ambas as partes
    text_top = tesseract_ocr(top)
    text_bottom = tesseract_ocr(bottom)

    # Junta o texto
    full_text = text_top + "\n" + text_bottom

    print(f"\n🧾 Página {page_index + 1}:")
    print(full_text)
    print("-" * 80)

    return {"page": page_index + 1, "text": full_text}

if __name__ == "__main__":
    pdf_path = "./data/to_process/08663.pdf"

    pdf = pdfium.PdfDocument(pdf_path)
    n_pages = len(pdf)

    print(f"📘 Total de páginas: {n_pages}")

    for page_index in range(n_pages):
        process_page({"pdf_path": pdf_path, "page_index": page_index})
