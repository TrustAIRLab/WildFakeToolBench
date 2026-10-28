from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class SyntheticPrnuInjectionTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="synthetic_prnu_injection",
            description="Injects a synthetic multiplicative sensor-noise pattern into images, mimicking the unique PRNU noise found in real camera photos to enhance authenticity."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        noise_intensity = params.get("noise_intensity", 0.5)
        pattern_type = params.get("pattern_type", "Gaussian")

        # Convert image to numpy array
        img_array = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        
        # Generate synthetic noise pattern
        if pattern_type == "Gaussian":
            noise = np.random.normal(loc=1.0, scale=noise_intensity, size=img_array.shape)
        elif pattern_type == "Poisson":
            noise = np.random.poisson(lam=1.0 + noise_intensity, size=img_array.shape) / 255.0
        else:
            raise ValueError("Unsupported pattern type. Use 'Gaussian' or 'Poisson'.")

        # Apply noise to the image
        noisy_image_array = img_array * noise
        
        # Clip values to stay within [0, 1] range
        noisy_image_array = np.clip(noisy_image_array, 0, 1)

        # Convert back to PIL Image
        noisy_image = Image.fromarray((noisy_image_array * 255).astype(np.uint8), mode="RGB")
        return noisy_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "noise_intensity": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Intensity of the noise pattern to be injected.",
                "default": 0.5,
                "required": False
            },
            "pattern_type": {
                "type": "str",
                "description": "Type of noise pattern to use ('Gaussian' or 'Poisson').",
                "default": "Gaussian",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Noise intensity: 0.0 to 1.0; Pattern type: 'Gaussian' or 'Poisson'. Use to enhance AI-generated images."