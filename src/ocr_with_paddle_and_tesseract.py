import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, List

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image
import pytesseract

# =========================
# CONFIGURAÇÕES GERAIS
# =========================

LANG = "por+eng"           # idiomas para OCR
DPI = 300                  # ~300 dpi é um bom compromisso p/ Tesseract
MAX_WORKERS = max(1, os.cpu_count() - 1)
USE_PADDLE = True          # tenta GPU (se PaddleOCR com CUDA estiver disponível)

# Tesseract: limite threads internas para evitar briga de threads
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
# Se tiver tessdata custom:
# os.environ["TESSDATA_PREFIX"] = "/caminho/para/tessdata"

# =========================
# PaddleOCR (opcional GPU)
# =========================
paddle_ocr = None
if USE_PADDLE:
    try:
        from paddleocr import PaddleOCR
        # use_gpu=True tenta usar CUDA
        paddle_ocr = PaddleOCR(
            use_gpu=True,
            lang="pt",          # ajuste se quiser multilíngue forte
            use_angle_cls=True,
            rec=True,
            det=True
        )
    except Exception:
        paddle_ocr = None  # se falhar, caímos no Tesseract (CPU)


# =========================
# Funções utilitárias
# =========================
def _preprocess_for_tesseract(img: np.ndarray) -> np.ndarray:
    """
    Pré-processamento leve para Tesseract (binarização etc.).
    Bom para documentos escaneados com texto impresso.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Binarização Otsu
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return bw


def _render_pdf_page(pdf: pdfium.PdfDocument, index: int, dpi: int = DPI) -> Image.Image:
    """
    Renderiza 1 página do PDF em PIL.Image usando pypdfium2 (rápido).
    """
    page = pdf[index]
    scale = dpi / 72.0  # 72 dpi é baseline do PDF
    bitmap = page.render(scale=scale).to_pil()
    return bitmap


def _pil_to_bgr(img_pil: Image.Image) -> np.ndarray:
    """
    Converte PIL -> ndarray BGR (OpenCV-style).
    """
    return cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def ocr_page_paddle(image_bgr: np.ndarray) -> str:
    """
    OCR via PaddleOCR (GPU se disponível).
    """
    import tempfile
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        Image.fromarray(image_rgb).save(tmp.name)
        result = paddle_ocr.ocr(tmp.name, cls=True)

    lines = []
    for page_res in result:
        for box, (text, conf) in page_res:
            lines.append(text)
    return "\n".join(lines)


def ocr_page_tesseract(image_bgr: np.ndarray) -> str:
    """
    OCR via Tesseract (CPU).
    Usa config mais rígida (--psm 6) para blocos tipo formulário/texto alinhado.
    """
    img = _preprocess_for_tesseract(image_bgr)
    cfg = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
    text = pytesseract.image_to_string(img, lang=LANG, config=cfg)
    return text


def process_page(job: Dict) -> Dict:
    """
    Função executada em subprocess:
    - renderiza a página
    - roda OCR (Paddle com GPU se disponível globalmente / fallback Tesseract)
    Retorna {page: idx, text: str}
    """
    pdf_path = job["pdf_path"]
    page_index = job["page_index"]

    # Cada processo precisa reabrir o PDF localmente
    pdf = pdfium.PdfDocument(pdf_path)
    pil_img = _render_pdf_page(pdf, page_index, DPI)
    bgr = _pil_to_bgr(pil_img)

    # Tentar PaddleOCR
    if paddle_ocr is not None:
        try:
            txt = ocr_page_paddle(bgr)
        except Exception:
            txt = ocr_page_tesseract(bgr)
    else:
        txt = ocr_page_tesseract(bgr)

    return {"page": page_index, "text": txt}


def ocr_pdf(pdf_path: Path,
            out_txt: Optional[Path] = None,
            out_json: Optional[Path] = None) -> Dict[int, str]:
    """
    Roda OCR no PDF inteiro em paralelo (1 processo por página).
    Salva:
      - TXT concatenado
      - JSON com [{"page": i, "text": "..."}]
    Retorna dict {pagina: texto}
    """
    pdf = pdfium.PdfDocument(str(pdf_path))
    n_pages = len(pdf)

    # Prepara jobs
    jobs = [{"pdf_path": str(pdf_path), "page_index": i} for i in range(n_pages)]
    page_texts: Dict[int, str] = {}

    # Paraleliza por página
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_page, job): job for job in jobs}
        for fut in as_completed(futures):
            res = fut.result()
            page_texts[res["page"]] = res["text"]

    # Ordenar páginas numericamente
    ordered_dict = {k: page_texts[k] for k in sorted(page_texts)}

    # Montar representação na forma pedida:
    # [
    #   {"page": 1, "text": "..."},
    #   {"page": 2, "text": "..."},
    # ]
    ordered_list: List[Dict[str, str]] = []
    for page_idx in sorted(ordered_dict.keys()):
        # page numerada a partir de 1 no JSON (como você mostrou)
        ordered_list.append({
            "page": page_idx + 1,
            "text": ordered_dict[page_idx],
        })

    # Gravar TXT concatenado (todas as páginas em sequência)
    if out_txt is not None:
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        with open(out_txt, "w", encoding="utf-8") as f:
            for page_obj in ordered_list:
                f.write(page_obj["text"] + "\n")

    # Gravar JSON na nova estrutura pedida
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(ordered_list, f, ensure_ascii=False, indent=2)

    # Continua retornando {pagina_idx: texto} em zero-based internamente
    return ordered_dict


if __name__ == "__main__":
    # Pastas fixas
    BASE_IN_DIR = Path("./data/to_process")
    BASE_OUT_DIR = Path("./data/output")

    BASE_IN_DIR.mkdir(parents=True, exist_ok=True)
    BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Loop em todos os arquivos da pasta de entrada
    for filename in os.listdir(BASE_IN_DIR):
        file_path = BASE_IN_DIR / filename

        # pula se não for arquivo normal ou não for .pdf
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in [".pdf"]:
            continue

        # nomes de saída específicos pra cada PDF
        base_name = file_path.stem  # "documento123" se "documento123.pdf"
        out_txt = BASE_OUT_DIR / f"{base_name}.txt"
        out_json = BASE_OUT_DIR / f"{base_name}.json"

        # roda o OCR
        result = ocr_pdf(
            pdf_path=file_path,
            out_txt=out_txt,
            out_json=out_json
        )

        print(f"[OK] OCR finalizado para {filename}")
        print(f" - Páginas processadas: {len(result)}")
        print(f" - TXT salvo em:  {out_txt}")
        print(f" - JSON salvo em: {out_json}")
