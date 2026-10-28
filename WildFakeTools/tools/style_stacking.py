from typing import Dict, Any
from PIL import Image, ImageEnhance
import numpy as np
from .base_tool import BaseTool


class StyleStackingTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="style_stacking",
            description="This tool enhances the realism of an image by applying multiple LoRA effects in a balanced manner. It allows for nuanced adjustments without overwhelming the image, ensuring it blends well with real photographs."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        max_strength = params.get("max_strength", 1.0)
        num_loras = params.get("num_loras", 1)

        # Ensure the parameters are within the specified limits
        max_strength = min(max(max_strength, 0.0), 2.0)
        num_loras = min(max(num_loras, 1), 3)

        # Convert image to RGB if not already
        image = image.convert("RGB")

        # Create a new image to apply the effects
        enhanced_image = Image.new("RGB", image.size)

        # Apply the LoRA effects based on the number of LoRAs and their combined strength
        for i in range(num_loras):
            strength = max_strength / num_loras
            enhancer = ImageEnhance.Color(image)
            enhanced_image = Image.blend(enhanced_image, enhancer.enhance(1 + strength), strength)

        return enhanced_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "max_strength": {
                "type": "float",
                "min": 0.0,
                "max": 2.0,
                "description": "Maximum strength of the LoRA effects combined.",
                "default": 1.0,
                "required": False
            },
            "num_loras": {
                "type": "int",
                "min": 1,
                "max": 3,
                "description": "Number of LoRAs to stack for the effect.",
                "default": 1,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Max strength: 0.0-2.0; Number of LoRAs: 1-3. Use this tool for subtle enhancements to blend AI-generated images with real photographs."