from io import BytesIO
from torch import autocast
import requests
import PIL
import torch
from diffusers import DiffusionPipeline
import gradio as gr


device = "cuda"
pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    #"CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=True
).to(device)

def predict(img_dict, prompt):
  init_img =  img_dict['image'].convert("RGB").resize((512, 512))
  mask_img = img_dict['mask'].convert("RGB").resize((512, 512))

  output = pipe(prompt=prompt, image=init_img, mask_image=mask_img, strength=0.75)

  return(output.images[0])


gr.Interface(
    predict,
    title = 'FreshFits by Pixlr (Stable Diffusion In-Painting Tool) by Kevin',
    description='Use this tool to look your best copping the freshest fits. You can edit images by uploading and sketching on the area to revamp, then the software generates new images based on your prompt.',
    inputs=[
        gr.Image(label='Upload or Edit Image',source = 'upload', tool = 'sketch', type = 'pil'),
        gr.Textbox(label='Outfit Description')
    ],
    outputs = [
        gr.Image(label='Generated Image')
        ]
).launch(debug = True)
