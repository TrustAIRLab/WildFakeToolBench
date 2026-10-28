from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class GaussianNoiseSimulationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="gaussian_noise_simulation",
            description="Apply Gaussian noise to the image to simulate sensor artifacts. Higher noise levels introduce realism by mimicking the imperfections found in photographs taken with real cameras."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        std_dev = params.get("std_dev", 0.1)
        apply_to = params.get("apply_to", True)

        if apply_to:
            # Convert image to numpy array
            img_array = np.array(image)
            # Generate Gaussian noise
            noise = np.random.normal(0, std_dev * 255, img_array.shape).astype(np.uint8)
            # Add noise to the image
            noisy_img_array = np.clip(img_array + noise, 0, 255)
            # Convert back to PIL Image
            noisy_image = Image.fromarray(noisy_img_array.astype('uint8'), 'RGB')
            return noisy_image
        else:
            return image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "std_dev": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Standard deviation of the Gaussian noise to apply.",
                "default": 0.1,
                "required": False
            },
            "apply_to": {
                "type": "bool",
                "description": "Whether to apply Gaussian noise to the image.",
                "default": True,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "std_dev: 0.0 to 1.0; apply_to: true/false. Use this tool to add realism to AI-generated images."