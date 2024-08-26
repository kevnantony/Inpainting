# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
from io import BytesIO
import torch
from diffusers import DiffusionPipeline

app = FastAPI()

# Initialize the model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=True
).to(device)

@app.post("/generate")
async def generate(prompt: str, image: UploadFile = File(...), mask: UploadFile = File(...)):
    try:
        # Convert uploaded files to PIL Images
        image = Image.open(BytesIO(await image.read())).convert("RGB").resize((512, 512))
        mask = Image.open(BytesIO(await mask.read())).convert("RGB").resize((512, 512))

        # Perform inpainting
        with torch.autocast(device):
            output = pipe(prompt=prompt, image=image, mask_image=mask, strength=0.75)

        # Convert result image to BytesIO
        buffered = BytesIO()
        output.images[0].save(buffered, format="PNG")
        buffered.seek(0)

        return StreamingResponse(buffered, media_type="image/png")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
