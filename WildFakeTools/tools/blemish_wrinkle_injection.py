from typing import Dict, Any
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from .base_tool import BaseTool


class BlemishWrinkleInjectionTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="blemish_wrinkle_injection",
            description="This tool adds natural imperfections to an image, such as blemishes or wrinkles, to make it appear more realistic and less perfect. This can help in evading detection by fake-image detectors."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        imperfection_type = params.get("imperfection_type", "blemishes")
        intensity = params.get("intensity", 0.5)

        # Convert image to RGB if needed
        image = image.convert("RGB")
        
        # Create a noise layer based on the imperfection type
        noise = np.random.normal(0, intensity * 25, (image.height, image.width, 3)).astype(np.uint8)
        
        # Apply noise to the image
        noisy_image = np.array(image) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        result_image = Image.fromarray(noisy_image)

        # Optionally apply a blur to simulate depth of field or imperfections
        if imperfection_type == "wrinkles":
            result_image = result_image.filter(ImageFilter.GaussianBlur(radius=intensity * 2))

        return result_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "imperfection_type": {
                "type": "str",
                "description": "Type of imperfection to add (e.g., blemishes, wrinkles).",
                "default": "blemishes",
                "required": False
            },
            "intensity": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Intensity of the imperfection effect.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Intensity ranges from 0.0 to 1.0; use this tool to add realistic imperfections to AI-generated images."