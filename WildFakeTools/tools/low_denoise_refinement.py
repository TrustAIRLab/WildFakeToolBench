from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class LowDenoiseRefinementTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="low_denoise_refinement",
            description="This tool applies a low-denoise filter to enhance details in AI-generated images while preserving the original composition."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        denoise_strength = params.get("denoise_strength", 0.5)
        iterations = params.get("iterations", 3)

        # Convert image to RGB if not already in that mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert image to numpy array
        img_array = np.array(image)

        # Apply a low-pass filter iteratively to simulate low denoise effect
        for _ in range(iterations):
            img_array = np.clip(img_array + (denoise_strength * (np.mean(img_array, axis=2, keepdims=True) - img_array)), 0, 255)

        # Convert back to PIL Image
        refined_image = Image.fromarray(np.uint8(img_array))

        return refined_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "denoise_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Strength of the denoise effect, where 0 is no effect and 1 is maximum denoise.",
                "default": 0.5,
                "required": False
            },
            "iterations": {
                "type": "int",
                "min": 1,
                "max": 5,
                "description": "Number of iterations to apply the denoise effect.",
                "default": 3,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Denoise strength: 0-1, Iterations: 1-5; useful for refining AI-generated images."