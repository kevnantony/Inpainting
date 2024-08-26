# gradio.py
import gradio as gr
import requests
from PIL import Image
import io

API_URL = "http://localhost:8000/generate"

def predict(init_img, mask_img, prompt):
    try:
        # Convert images to bytes
        init_img_bytes = io.BytesIO()
        init_img.save(init_img_bytes, format='PNG')
        init_img_bytes.seek(0)

        mask_img_bytes = io.BytesIO()
        mask_img.save(mask_img_bytes, format='PNG')
        mask_img_bytes.seek(0)

        # Prepare the payload
        files = {
            "image": ("image.png", init_img_bytes, "image/png"),
            "mask": ("mask.png", mask_img_bytes, "image/png")
        }
        data = {"prompt": prompt}

        # Call the API
        response = requests.post(API_URL, files=files, data=data)
        response.raise_for_status()  # Raise an error for bad responses

        # Check if response contains an image
        if 'image' in response.headers['Content-Type']:
            img_data = io.BytesIO(response.content)
            img = Image.open(img_data)
            return img
        else:
            return "Error: Unexpected content type"

    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

# Set up the Gradio interface
gr.Interface(
    fn=predict,
    title="FreshFits by Pixlr (Stable Diffusion In-Painting Tool) by Kevin",
    description="Use this tool to look your best copping the freshest fits. Upload an image and a mask to edit specific areas, then the software generates new images based on your prompt.",
    inputs=[
        gr.Image(label='Upload Image', type='pil'),
        gr.Image(label='Upload Mask', type='pil'),
        gr.Textbox(label='Outfit Description')
    ],
    outputs=gr.Image(label='Generated Image')
).launch(debug=True)
