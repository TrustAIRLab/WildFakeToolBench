from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class SubtlePerPixelPerturbationsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="subtle_per_pixel_perturbations",
            description="Applies extremely subtle structured per-pixel perturbations to interfere with watermark detection. These perturbations keep the image visually unchanged while disrupting the watermark's detection features.",
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        perturbation_strength = params.get("perturbation_strength", 0.01)
        pattern_type = params.get("pattern_type", "Gaussian")

        image = image.convert("RGB")
        image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]

        if pattern_type == "Gaussian":
            noise = np.random.normal(0, perturbation_strength, image_array.shape)
        elif pattern_type == "Uniform":
            noise = np.random.uniform(-perturbation_strength, perturbation_strength, image_array.shape)
        else:
            raise ValueError("Invalid pattern_type. Choose 'Gaussian' or 'Uniform'.")

        perturbed_image_array = np.clip(image_array + noise, 0, 1)  # Ensure values are in [0, 1]
        perturbed_image = Image.fromarray((perturbed_image_array * 255).astype(np.uint8), mode="RGB")

        return perturbed_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "perturbation_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Strength of the perturbation applied to each pixel.",
                "default": 0.01,
                "required": False
            },
            "pattern_type": {
                "type": "str",
                "description": "Type of noise pattern to apply ('Gaussian' or 'Uniform').",
                "default": "Gaussian",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Perturbation strength: 0.0 to 1.0; pattern type: 'Gaussian' or 'Uniform'. Useful for subtly altering images without visible changes."