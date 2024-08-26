import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

class InpaintingModel:
    def __init__(self, model_id="stabilityai/stable-diffusion-2-inpainting"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            revision="fp16",
            use_auth_token=True
        ).to(self.device)
        self.pipe.enable_xformers_memory_efficient_attention()        
    
    def inpaint(self, prompt, image, mask, guidance_scale, num_inference_steps, strength):
        with torch.autocast("cuda"):
            result=self.pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                strength=strength
            ).images[0]
        return result
