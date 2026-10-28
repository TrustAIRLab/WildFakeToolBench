from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class SkinRealismCorrectionTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="skin_realism_correction",
            description="This tool enhances the realism of skin textures by simulating undesirable traits, resulting in more natural skin appearances."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        negative_traits = params.get("negative_traits", ["Zombie", "Sick"])
        
        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert image to numpy array for processing
        img_array = np.array(image)
        
        # Apply a Gaussian blur and a slight color shift.
        blurred = Image.fromarray(img_array).filter(ImageFilter.GaussianBlur(radius=2))
        img_array_blurred = np.array(blurred)
        
        # Create a color shift effect to simulate undesirable skin tones
        color_shift = np.array([0.9, 0.8, 0.7])  # Slightly desaturate colors
        img_array_realistic = img_array_blurred * color_shift
        
        # Clip values to ensure they remain valid pixel values
        img_array_realistic = np.clip(img_array_realistic, 0, 255).astype(np.uint8)
        
        # Return the modified image as a new PIL Image
        return Image.fromarray(img_array_realistic, mode='RGB')

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "negative_traits": {
                "type": "list",
                "description": "List of negative traits to simulate undesirable skin features.",
                "default": ["Zombie", "Sick"],
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Use with a variety of negative traits; default traits are 'Zombie' and 'Sick'."