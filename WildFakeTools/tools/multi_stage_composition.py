from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class MultiStageCompositionTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="multi_stage_composition",
            description="This tool enhances images by applying a series of artistic transformations, mimicking human artistry to create a more authentic aesthetic."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        model_selection = params.get("model_selection", ["contrast", "blur", "sharpen"])
        refinement_steps = params.get("refinement_steps", 3)

        # Convert image to RGB if not already
        image = image.convert("RGB")
        img_array = np.array(image)

        # Apply selected transformations
        for _ in range(refinement_steps):
            for model in model_selection:
                if model == "contrast":
                    img_array = self.adjust_contrast(img_array)
                elif model == "blur":
                    img_array = self.apply_blur(img_array)
                elif model == "sharpen":
                    img_array = self.apply_sharpen(img_array)

        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def adjust_contrast(self, img_array: np.ndarray) -> np.ndarray:
        factor = 1.5  # Increase contrast
        mean = np.mean(img_array, axis=(0, 1), keepdims=True)
        return (img_array - mean) * factor + mean

    def apply_blur(self, img_array: np.ndarray) -> np.ndarray:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(img_array, sigma=1)

    def apply_sharpen(self, img_array: np.ndarray) -> np.ndarray:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        from scipy.signal import convolve2d
        for i in range(3):  # Apply to each channel
            img_array[..., i] = convolve2d(img_array[..., i], kernel, mode='same', boundary='wrap')
        return img_array

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "model_selection": {
                "type": "list",
                "description": "List of models to apply (e.g., contrast, blur, sharpen).",
                "default": ["contrast", "blur", "sharpen"],
                "required": False
            },
            "refinement_steps": {
                "type": "int",
                "min": 1,
                "max": 10,
                "description": "Number of refinement steps to apply.",
                "default": 3,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Parameters: model_selection (e.g., contrast, blur, sharpen), refinement_steps (1-10). Use for enhancing artistic quality."