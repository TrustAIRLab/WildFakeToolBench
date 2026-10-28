from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class FftPhasePerturbationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="fft_phase_perturbation",
            description="Introduce phase perturbation in the Fourier transform domain to simulate non-uniform distortions.",
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        perturbation_strength = params.get("perturbation_strength", 0.5)
        frequency_range = params.get("frequency_range", 1.0)

        # Convert image to RGB and then to numpy array
        image = image.convert("RGB")
        image_array = np.array(image)

        # Perform FFT and shift the zero frequency component to the center
        f_transform = np.fft.fft2(image_array)
        f_transform_shifted = np.fft.fftshift(f_transform)

        # Get the phase and magnitude
        magnitude = np.abs(f_transform_shifted)
        phase = np.angle(f_transform_shifted)

        # Create a random perturbation in the phase
        random_perturbation = np.random.uniform(-perturbation_strength, perturbation_strength, phase.shape)
        perturbed_phase = phase + random_perturbation

        # Create the new complex array with the perturbed phase
        perturbed_f_transform = magnitude * np.exp(1j * perturbed_phase)

        # Inverse FFT to get the modified image
        f_transform_ishifted = np.fft.ifftshift(perturbed_f_transform)
        modified_image_array = np.fft.ifft2(f_transform_ishifted).real

        # Clip values to be in the valid range and convert back to uint8
        modified_image_array = np.clip(modified_image_array, 0, 255).astype(np.uint8)

        # Return as a PIL Image
        return Image.fromarray(modified_image_array, mode="RGB")

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "perturbation_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Strength of the phase perturbation.",
                "default": 0.5,
                "required": False
            },
            "frequency_range": {
                "type": "float",
                "min": 0.1,
                "max": 10.0,
                "description": "Range of frequencies to apply perturbation.",
                "default": 1.0,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Perturbation strength (0.0 to 1.0) and frequency range (0.1 to 10.0) can be adjusted for desired effects."