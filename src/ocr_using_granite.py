from chandra.model.hf import generate_hf
from chandra.model.schema import BatchInputItem
from transformers import AutoModel  # ou conforme especificado
from PIL import Image

model = AutoModel.from_pretrained("datalab-to/chandra").cuda()  # ou no seu device
image = Image.open("path_to_document.jpg").convert("RGB")
batch = [ BatchInputItem(image=image, prompt_type="ocr_layout") ]
result = generate_hf(batch, model)[0]
markdown = result.markdown  # ou result.json ou result.html conforme saída
print(markdown)
