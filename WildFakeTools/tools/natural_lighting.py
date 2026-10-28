from typing import Dict, Any
from PIL import Image, ImageEnhance
import numpy as np
from .base_tool import BaseTool


class NaturalLightingTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="natural_lighting",
            description="This tool enhances the realism of an image by simulating natural lighting effects, such as soft shadows and diffused light, to reduce overly dramatic or polished appearances."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        lighting_description = params.get("lighting_description", "soft")
        
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert image to numpy array for manipulation
        img_array = np.array(image)
        
        # Apply natural lighting effects based on the description
        if lighting_description == "soft":
            img_array = self.apply_soft_lighting(img_array)
        elif lighting_description == "bright":
            img_array = self.apply_bright_lighting(img_array)
        elif lighting_description == "dim":
            img_array = self.apply_dim_lighting(img_array)
        
        # Convert back to PIL Image
        return Image.fromarray(np.clip(img_array, 0, 255).astype('uint8'))

    def apply_soft_lighting(self, img_array: np.ndarray) -> np.ndarray:
        # Simulate soft lighting by applying a Gaussian blur and enhancing brightness
        blurred = np.clip(img_array.astype(np.float32) + np.random.normal(0, 5, img_array.shape), 0, 255)
        return blurred

    def apply_bright_lighting(self, img_array: np.ndarray) -> np.ndarray:
        # Increase brightness
        enhancer = ImageEnhance.Brightness(Image.fromarray(img_array))
        return np.array(enhancer.enhance(1.5))

    def apply_dim_lighting(self, img_array: np.ndarray) -> np.ndarray:
        # Decrease brightness
        enhancer = ImageEnhance.Brightness(Image.fromarray(img_array))
        return np.array(enhancer.enhance(0.7))

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "lighting_description": {
                "type": "str",
                "description": "Description of the lighting effect to apply (e.g., 'soft', 'bright', 'dim').",
                "default": "soft",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Lighting descriptions: 'soft', 'bright', 'dim' for varying levels of realism."