# Inpainting Project

This project implements an image inpainting service using FastAPI and the `StableDiffusionInpaintPipeline` from the Hugging Face Diffusers library. The service allows users to perform inpainting on images by providing a prompt, an image, and a mask.

## Features

- **Image Inpainting**: Fill in missing parts of images based on a given prompt using stable diffusion.
- **FastAPI**: A modern, fast (high-performance) web framework for building APIs with Python 3.7+.
- **Ray Serve**: Scalable and flexible deployment of the FastAPI service.

## Prerequisites

- **Python 3.12.5**
- **pip** (Python package installer)

## Download the pretrained model
- **stabilityai/stable-diffusion-2-inpainting** 
```bash
from diffusers import StableDiffusionInpaintPipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
```
## Run the fast api server
```bash
python app.py
```
## API Endpoints
- **Health Check**
- URL: /
- Method: GET
- Description: A simple endpoint to check if the service is running.
- Response: {"message": "Welcome"}

- **Generate Inpainting**
- URL: /generate
- Method: POST
- Description: Generate an inpainted image based on the provided prompt, image, and mask.
- Parameters:
- - prompt (str): A text prompt describing the desired completion.
- - image (file): The image file to be inpainted.
- - mask (file): The mask image where white indicates the area to be inpainted.
- - guidance_scale (float, optional): The scale for classifier-free guidance.
- - num_inference_steps (int, optional): Number of denoising steps.
- - strength (float, optional): Strength of the initial image preservation.

## Example Usage
- Run the service and use a tool like curl or Postman to send a request to the /generate endpoint.
- View the result: The inpainted image will be saved locally at ./saved_inpainting_result.png and can also be accessed as a base64 string in the response.