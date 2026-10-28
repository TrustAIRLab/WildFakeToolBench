from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class PatternObscuringRelaunderTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="pattern_obscuring_relaunder",
            description="Applies noise and distortion to images, enhancing their realism by obscuring original patterns."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        processing_strength = params.get("processing_strength", 0.5)
        
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert image to numpy array
        img_array = np.array(image)

        # Apply Gaussian noise
        noise = np.random.normal(0, processing_strength * 25, img_array.shape).astype(np.uint8)
        noisy_image = np.clip(img_array + noise, 0, 255)

        # Convert back to PIL Image
        result_image = Image.fromarray(noisy_image)

        # Apply a slight blur to further obscure details
        result_image = result_image.filter(ImageFilter.GaussianBlur(radius=1 * processing_strength))

        return result_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "processing_strength": {
                "type": "float",
                "min": 0,
                "max": 1,
                "description": "Strength of noise and distortion applied to the image.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Processing strength ranges from 0 (no effect) to 1 (maximum distortion); useful for enhancing realism."