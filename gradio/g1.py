import gradio as gr
import requests
from PIL import Image
import io

# URL of the FastAPI server
API_URL = "http://localhost:8000/predict/"

def predict(image, mask, prompt):
    try:
        # Convert images to bytes
        image_buffered = io.BytesIO()
        image.save(image_buffered, format="PNG")
        image_bytes = image_buffered.getvalue()

        mask_buffered = io.BytesIO()
        mask.save(mask_buffered, format="PNG")
        mask_bytes = mask_buffered.getvalue()

        # Prepare the payload for the API request
        files = {
            'image': ('image.png', image_bytes, 'image/png'),
            'mask': ('mask.png', mask_bytes, 'image/png')
        }
        data = {
            'prompt': prompt
        }

        # Call the FastAPI endpoint
        response = requests.post(API_URL, files=files, data=data)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            return Image.open(io.BytesIO(result['image']))
        else:
            # Return an error image or message
            error_message = response.json().get('detail', 'Error: Unable to get result from the FastAPI server.')
            return gr.Error(error_message)
    except Exception as e:
        return gr.Error(f"An unexpected error occurred: {str(e)}")

# Define Gradio interface with mask editing capabilities
gr.Interface(
    fn=predict,
    title='FreshFits by Pixlr (Stable Diffusion In-Painting Tool)',
    description='Upload an image, create or edit a mask using the built-in editor, and provide a description to generate a new image based on your prompt.',
    inputs=[
        gr.Image(label='Upload or Edit Image', source='upload', tool='editor', type='pil'),
        gr.Image(label='Create or Edit Mask', source='upload', tool='editor', type='pil'),
        gr.Textbox(label='Outfit Description', placeholder='Describe the outfit or changes you want to see')
    ],
    outputs=[
        gr.Image(label='Generated Image')
    ]
).launch(debug=True)
