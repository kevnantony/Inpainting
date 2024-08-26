# from diffusers import StableDiffusionInpaintPipeline
# pipe = StableDiffusionInpaintPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2-inpainting",
#     torch_dtype=torch.float16,
# )
# pipe.to("cuda")
# prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
# #image and mask_image should be PIL images.
# #The mask structure is white for inpainting and black for keeping as is
# image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
# image.save("./yellow_cat_on_park_bench.png")


from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

# Initialize the pipeline with the correct model and data type
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)

# Check for CUDA availability and move the model to GPU if available
if torch.cuda.is_available():
    pipe.to("cuda")
else:
    print("CUDA is not available. The model will run on CPU.")
    pipe.to("cpu")

# Define the prompt for inpainting
prompt = "Face of a yellow cat, high resolution, sitting on a park bench"

# Load your images (ensure these are PIL images)
# Replace 'path_to_image' and 'path_to_mask' with actual file paths
image = Image.open('/home/ubuntu/inpainting_project/f1.png').convert("RGB")
mask_image = Image.open('/home/ubuntu/inpainting_project/f2.png').convert("RGB")  # Ensure mask_image is in the correct format

try:
    # Perform inpainting
    result = pipe(prompt=prompt, image=image, mask_image=mask_image)
    # Save the resulting image
    result.images[0].save("./yellow_cat_on_park_bench.png")
    print("Inpainting complete. Image saved as 'yellow_cat_on_park_bench.png'.")
except Exception as e:
    print(f"An error occurred during inpainting: {e}")
