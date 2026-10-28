from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class MotionBlurEffectTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="motion_blur_effect",
            description="Introduce slight motion blur to simulate unintentional camera shake in casual shots."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        blur_strength = params.get("blur_strength", 0.1)
        direction = params.get("direction", "horizontal")

        # Convert image to RGB if not already
        image = image.convert("RGB")

        # Create a motion blur kernel
        kernel_size = int(blur_strength * 20) + 1  # Scale blur strength to kernel size
        if kernel_size % 2 == 0:  # Ensure kernel size is odd
            kernel_size += 1
        
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        if direction == "horizontal":
            kernel[int(kernel_size / 2), :] = np.ones(kernel_size)
        elif direction == "vertical":
            kernel[:, int(kernel_size / 2)] = np.ones(kernel_size)
        else:
            raise ValueError("Direction must be 'horizontal' or 'vertical'.")

        kernel /= kernel.sum()  # Normalize the kernel

        # Apply the blur using the kernel. PIL's ImageFilter.Kernel only
        # supports fixed 3x3/5x5 sizes, but kernel_size scales up to 21 here,
        # so convolve directly instead.
        from scipy.ndimage import convolve
        img_array = np.array(image, dtype=np.float32)
        blurred_array = np.stack(
            [convolve(img_array[..., c], kernel, mode="reflect") for c in range(img_array.shape[-1])],
            axis=-1,
        )
        blurred_image = Image.fromarray(np.clip(blurred_array, 0, 255).astype(np.uint8))

        return blurred_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "blur_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Strength of the motion blur effect.",
                "default": 0.1,
                "required": False
            },
            "direction": {
                "type": "str",
                "description": "Direction of the motion blur ('horizontal' or 'vertical').",
                "default": "horizontal",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Adjust blur_strength (0.0-1.0) for intensity; direction can be 'horizontal' or 'vertical'."