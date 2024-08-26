import sys
sys.path.append("../src")

from fastapi import FastAPI, UploadFile, File
import ray
from ray import serve
from PIL import Image
from io import BytesIO
import base64
import torch
from diffusers import DiffusionPipeline
from pydantic import BaseModel
from typing import Optional
import io

app = FastAPI()

# Initialize the InpaintingInterface class
class InpaintingInterface:
    def __init__(self):
        try:
            # Initialize the pipeline
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting",torch_dtype=torch.float16,)

            # Check for CUDA availability and move the model to GPU if available
            if torch.cuda.is_available():
                self.pipe.to("cuda")
            else:
                print("CUDA is not available. The model will run on CPU.")
                self.pipe.to("cpu")

        except RuntimeError as e:
            print(f"RuntimeError during model initialization: {e}")
            raise e

        except Exception as e:
            print(f"Exception during model initialization: {e}")
            raise e

    def run(self, prompt: str, image: Image.Image, mask: Image.Image, guidance_scale: float, num_inference_steps: int, strength: float) -> Image.Image:
        result = self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            strength=strength
        ).images[0]
        return result
    
    def launch(self):
        iface = gr.Interface(
            fn=self.run,
            inputs=[
                gr.Textbox(label="Prompt"),
                gr.Image(type="pil", label="Image"),
                gr.Image(type="pil", label="Mask"),
                gr.Slider(5.0, 20.0, step=0.5, label="Guidance Scale"),
                gr.Slider(20, 100, step=1, label="Inference Steps"),
                gr.Slider(0.5, 1.0, step=0.1, label="Strength")
            ],
            outputs="image",
            title="Stable Diffusion Inpainting"
        )
        iface.launch(share=True)

# Health check route
@app.get("/")
async def health_check():
    return {"message": "Welcome"}

# Ray Serve deployment for the first inpainting service
@serve.deployment
@serve.ingress(app)
class InpaintingService:
    def __init__(self):
        self.interface = InpaintingInterface()

    @app.post("/generate")
    async def generate(self, 
                       prompt: str, 
                       image: UploadFile = File('/home/ubuntu/inpainting_project/f1.png'), 
                       mask: UploadFile = File('/home/ubuntu/inpainting_project/f2.png'), 
                       guidance_scale: float = 7.5, 
                       num_inference_steps: int = 50, 
                       strength: float = 0.75):
        
        # Convert uploaded files to PIL Images
        image = Image.open(BytesIO(await image.read())).convert("RGB")
        mask = Image.open(BytesIO(await mask.read())).convert("RGB")
        
        # Perform inpainting
        result_image = self.interface.run(prompt, image, mask, guidance_scale, num_inference_steps, strength)
        
        # Convert the result to base64
        buffered = BytesIO()
        result_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        save_path = "/home/ubuntu/inpainting_project/Images/"
        result_image.save(save_path)
        print(f"Image saved to {save_path}")
        
        return {"image": img_str}
    

# Initialize the new DiffusionPipeline model
device = "cuda" if torch.cuda.is_available() else "cpu"
new_pipe = DiffusionPipeline.from_pretrained(
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
        output = new_pipe(prompt=prompt.prompt, image=init_img, mask_image=mask_img, strength=0.75)
    
    # Convert output to bytes
    output_image = output.images[0]
    buffer = BytesIO()
    output_image.save(buffer, format="PNG")
    buffer.seek(0)
    
    return {
        "image": buffer.getvalue()
    }

# Deploy the service with a route prefix
ray.init(ignore_reinit_error=True)
serve.start()
serve.run(InpaintingService.bind(), route_prefix="/inpainting")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
