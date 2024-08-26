import sys
sys.path.append("../src")

from fastapi import FastAPI, UploadFile, File
import ray
from ray import serve
from interface import InpaintingInterface
from PIL import Image
from io import BytesIO
import base64

app = FastAPI()

# Health check route
@app.get("/")
async def health_check():
    return {"message": "Welcome"}

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
    

# Deploy the service with a route prefix
ray.init(ignore_reinit_error=True)
serve.start()
serve.run(InpaintingService.bind(), route_prefix="/inpainting")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
