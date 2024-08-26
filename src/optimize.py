# import optuna
# from inpainting import InpaintingModel
# from skimage.metrics import structural_similarity as ssim
# import cv2
# import numpy as np

# class InpaintingOptimizer:
#     def __init__(self, prompt, image, mask):
#         self.prompt = prompt
#         self.image = image
#         self.mask = mask
#         self.model = InpaintingModel()

#     def objective(self, trial):
#         guidance_scale = trial.suggest_float("guidance_scale", 5.0, 20.0)
#         num_inference_steps = trial.suggest_int("num_inference_steps", 20, 100)
#         strength = trial.suggest_float("strength", 0.5, 1.0)

#         generated_image = self.model.inpaint(self.prompt, self.image, self.mask, guidance_scale, num_inference_steps, strength)

#         # Implement an actual evaluation function
#         score = self.evaluate_image(generated_image)
#         return score

#     def evaluate_image(self, generated_image):
#         # Placeholder evaluation function, replace with actual metric
#         return 1.0

#     def optimize(self, n_trials=50):
#         study = optuna.create_study(direction="maximize")
#         study.optimize(self.objective, n_trials=n_trials)
#         return study.best_params


import optuna
from inpainting import InpaintingModel
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np

class InpaintingOptimizer:
    def __init__(self, prompt, image, mask, target_image):
        self.prompt = prompt
        self.image = image
        self.mask = mask
        self.target_image = target_image  # Add target image for evaluation
        self.model = InpaintingModel()

    def objective(self, trial):
        guidance_scale = trial.suggest_float("guidance_scale", 5.0, 20.0)
        num_inference_steps = trial.suggest_int("num_inference_steps", 20, 100)
        strength = trial.suggest_float("strength", 0.5, 1.0)

        generated_image = self.model.inpaint(self.prompt, self.image, self.mask, guidance_scale, num_inference_steps, strength)

        # Implement an actual evaluation function
        score = self.evaluate_image(generated_image)
        return score

    def evaluate_image(self, generated_image):
        # Convert images to grayscale for SSIM
        generated_image_gray = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)
        target_image_gray = cv2.cvtColor(self.target_image, cv2.COLOR_BGR2GRAY)

        # Calculate SSIM
        ssim_score = ssim(target_image_gray, generated_image_gray)

        # Calculate PSNR
        psnr_score = cv2.PSNR(self.target_image, generated_image)

        # Combine scores (you can adjust the weights according to your needs)
        combined_score = (ssim_score + psnr_score) / 2
        return combined_score

    def optimize(self, n_trials=50):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_params
