from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import torch
from diffusers import DiffusionPipeline
import io

app = FastAPI()

# Initialize the model
device = "cuda"
pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=True
).to(device)

# Define the request body model
class PredictRequest(BaseModel):
    prompt: str

@app.post("/predict/")
async def predict(prompt: PredictRequest, image: UploadFile = File(...), mask: UploadFile = File(...)):
    # Load images
    init_img = Image.open(io.BytesIO(await image.read())).convert("RGB").resize((512, 512))
    mask_img = Image.open(io.BytesIO(await mask.read())).convert("RGB").resize((512, 512))
    
    # Perform inpainting
    with torch.no_grad():
        output = pipe(prompt=prompt.prompt, image=init_img, mask_image=mask_img, strength=0.75)
    
    # Convert output to bytes
    output_image = output.images[0]
    buffer = BytesIO()
    output_image.save(buffer, format="PNG")
    buffer.seek(0)
    
    return {
        "image": buffer.getvalue()
    }
