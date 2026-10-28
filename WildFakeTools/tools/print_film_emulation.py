from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class PrintFilmEmulationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="print_film_emulation",
            description="This tool applies print-film emulation to soften the appearance of images, mimicking the gentle highlight compression and tone response of analog prints."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        softening_strength = params.get("softening_strength", 0.5)
        
        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert image to numpy array
        img_array = np.array(image) / 255.0
        
        # Apply softening effect
        # Using a simple tone mapping function to simulate film emulation
        img_array = np.clip(img_array ** (1.0 - softening_strength), 0, 1)
        
        # Convert back to uint8
        img_array = (img_array * 255).astype(np.uint8)
        
        # Create a new PIL image from the modified array
        return Image.fromarray(img_array, 'RGB')

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "softening_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Controls the strength of the softening effect. Higher values result in a softer image.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "softening_strength: 0.0 to 1.0; use to reduce harshness in AI-generated images."