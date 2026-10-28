from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class ProgressiveRefinementPassTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="progressive_refinement_pass",
            description="This tool applies iterative refinement to enhance the realism of an image. It progressively sharpens and adjusts the image to create a more polished final output."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        iteration_count = params.get("iteration_count", 5)
        prompt_adjustment = params.get("prompt_adjustment", "")

        image = image.convert("RGB")
        for _ in range(iteration_count):
            image = self.refine_image(image, prompt_adjustment)

        return image

    def refine_image(self, image: Image.Image, prompt_adjustment: str) -> Image.Image:
        # Apply a slight sharpening filter to enhance details
        sharpened_image = image.filter(ImageFilter.SHARPEN)
        
        # Convert to numpy array for further processing if needed
        image_array = np.array(sharpened_image)

        # Scale brightness by the length of the adjustment string.
        if prompt_adjustment:
            adjustment_value = min(max(len(prompt_adjustment) / 100, 0), 1)
            image_array = np.clip(image_array * (1 + adjustment_value * 0.1), 0, 255).astype(np.uint8)

        return Image.fromarray(image_array)

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "iteration_count": {
                "type": "int",
                "min": 1,
                "max": 10,
                "description": "Number of refinement iterations to apply.",
                "default": 5,
                "required": False
            },
            "prompt_adjustment": {
                "type": "str",
                "description": "A string that influences the adjustment of the image.",
                "default": "",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "1-10 iterations; useful for gradually enhancing image realism."