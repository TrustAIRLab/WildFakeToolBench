from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class AiNormalizerTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="ai_normalizer",
            description="This tool subtly modifies the image to enhance its realism while maintaining visual fidelity, making it harder for detectors to recognize as AI-generated."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        modification_strength = params.get("modification_strength", 0.5)
        
        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert image to numpy array
        img_array = np.array(image)
        
        # Apply subtle adjustments
        noise = np.random.normal(loc=0, scale=modification_strength * 10, size=img_array.shape)
        modified_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        modified_image = Image.fromarray(modified_array, 'RGB')
        return modified_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "modification_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Strength of the modification applied to the image.",
                "default": 0.5,
                "required": False
            },
            "model_type": {
                "type": "str",
                "description": "Type of model used for normalization (e.g., LPIPS).",
                "default": "LPIPS",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Modification strength: 0.0 to 1.0; use when subtle adjustments are needed to enhance realism."