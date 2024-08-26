import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import gradio as gr

# class InpaintingInterface:
#     def __init__(self):
#         # Initialize the Stable Diffusion inpainting model
#         self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
#             "stabilityai/stable-diffusion-2-inpainting",
#             torch_dtype=torch.float16,
#         )
#         self.pipe.to("cuda")
class InpaintingInterface:
    def __init__(self):
        try:
            # Initialize the pipeline (this is a placeholder, replace with your actual pipeline initialization)
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting",torch_dtype=torch.float16,)
#   # Replace with your pipeline initialization code

            # Check for CUDA availability and move the model to GPU if available
            if torch.cuda.is_available():
                self.pipe.to("cuda")
            else:
                print("CUDA is not available. The model will run on CPU.")
                self.pipe.to("cpu")

        except RuntimeError as e:
            print(f"RuntimeError during model initialization: {e}")
            # Optionally, you can raise an exception or handle the error as needed
            raise e

        except Exception as e:
            print(f"Exception during model initialization: {e}")
            # Optionally, you can raise an exception or handle the error as needed
            raise e

    def run(self, prompt: str, image: Image.Image, mask: Image.Image, guidance_scale: float, num_inference_steps: int, strength: float) -> Image.Image:
        # Perform inpainting using the provided prompt, image, and mask
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