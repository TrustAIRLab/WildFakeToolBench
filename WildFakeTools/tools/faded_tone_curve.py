from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class FadedToneCurveTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="faded_tone_curve",
            description="Applies a faded tone curve to lift blacks and reduce contrast for a soft, vintage look."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        lift_amount = params.get("lift_amount", 0.2)
        contrast_reduction = params.get("contrast_reduction", 0.5)

        # Convert image to RGB and then to numpy array
        image = image.convert("RGB")
        img_array = np.array(image) / 255.0  # Normalize to [0, 1]

        # Apply faded tone curve
        # Lift blacks
        lifted = np.clip(img_array + lift_amount * (1 - img_array), 0, 1)

        # Reduce contrast
        mean = np.mean(lifted, axis=2, keepdims=True)
        contrast_adjusted = mean + contrast_reduction * (lifted - mean)

        # Clip values to [0, 1]
        final_image = np.clip(contrast_adjusted, 0, 1)

        # Convert back to uint8 and create a new PIL Image
        final_image = (final_image * 255).astype(np.uint8)
        return Image.fromarray(final_image, mode="RGB")

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "lift_amount": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Amount to lift the blacks in the image.",
                "default": 0.2,
                "required": False
            },
            "contrast_reduction": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Amount to reduce the contrast in the image.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Lift amount and contrast reduction should be between 0.0 and 1.0 for best results."