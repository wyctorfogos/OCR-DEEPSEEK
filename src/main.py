import fitz  # PyMuPDF
from PIL import Image
import io, os
from transformers import AutoModel, AutoTokenizer
import torch

# 1) Carregar modelo (igual ao snippet acima)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name = "deepseek-ai/DeepSeek-OCR"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation="flash_attention_2",
    trust_remote_code=True,
    use_safetensors=True
).eval().cuda().to(torch.bfloat16)

PROMPT = "<image>\n<|grounding|>Convert the document to markdown."

def page_to_pil(page, zoom=2.0):
    mat = fitz.Matrix(zoom, zoom)  # aumenta resolução
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def infer_image(pil_img, tmp_path="__tmp_page.png",
                base_size=1024, image_size=640, crop_mode=True,
                save_results=False, test_compress=True):
    pil_img.save(tmp_path)
    # API do modelo definida no card/README: model.infer(...)
    res = model.infer(
        tokenizer,
        prompt=PROMPT,
        image_file=tmp_path,
        output_path=".",
        base_size=base_size, image_size=image_size,
        crop_mode=crop_mode,
        save_results=save_results,
        test_compress=test_compress
    )
    return res

def ocr_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    outputs = []
    for i, page in enumerate(doc, start=1):
        img = page_to_pil(page, zoom=2.0)
        res = infer_image(img, tmp_path=f"__page_{i}.png")
        # A API retorna string/objeto dependendo da config; normalize:
        text = res if isinstance(res, str) else str(res)
        outputs.append(f"\n\n# Page {i}\n{text}")
    return "\n".join(outputs)

if __name__ == "__main__":
    pdf = "./data/to_process/CV- CIENTISTA_DE_DADOS-WYCTOR_FOGOS_DA_ROCHA.pdf"
    texto_md = ocr_pdf(pdf)
    with open("saida.md", "w", encoding="utf-8") as f:
        f.write(texto_md)
