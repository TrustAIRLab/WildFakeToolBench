from typing import Dict, Any, List
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class MultiGenerationBlendTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="multi_generation_blend",
            description="Mix elements from different generated images by layering and blending. This creates a more complex and less detectable image by introducing variability."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        # Read parameters
        layer_selection = params.get("layer_selection", [])
        blend_mode = params.get("blend_mode", "normal")

        # Convert the main image to RGB
        base_image = image.convert("RGB")
        base_array = np.array(base_image)

        # Load each layer image; paths that are not readable files are skipped.
        layers = []
        for layer_path in layer_selection:
            try:
                layer_image = Image.open(layer_path).convert("RGB")
                layers.append(np.array(layer_image.resize(base_image.size)))
            except (FileNotFoundError, OSError):
                continue

        if not layers:
            # No usable external layer: blend with a mirrored copy of the input.
            layers = [np.array(base_image.transpose(Image.FLIP_LEFT_RIGHT))]

        for layer_array in layers:
            # Blend images based on the selected blend mode
            if blend_mode == "normal":
                base_array = np.clip(base_array * 0.7 + layer_array * 0.3, 0, 255)
            elif blend_mode == "multiply":
                base_array = np.clip((base_array / 255.0) * (layer_array / 255.0) * 255, 0, 255)

        # Convert back to PIL Image
        result_image = Image.fromarray(base_array.astype('uint8'), 'RGB')
        return result_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "layer_selection": {
                "type": "list",
                "description": "List of image paths to blend with the base image.",
                "default": [],
                "required": False
            },
            "blend_mode": {
                "type": "str",
                "description": "Blend mode to use for combining images.",
                "default": "normal",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Blend multiple images using 'normal' or 'multiply' mode; layer_selection should contain paths to images."