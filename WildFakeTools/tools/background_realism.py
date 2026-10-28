"""
Background Realism Tool — reduce AI's perfect/uniform background appearance.

Source: Blog posts on making AI images look realistic; Reddit discussions on
AI images looking "too perfect" in backgrounds and environments.

AI-generated backgrounds are uniformly detailed, perfectly consistent, and free
of natural imperfections. Real photographs have: slight focus falloff at edges,
environmental dust/haze, motion blur on distant elements, irregular lighting
patches, and authentic depth cues. PIL cannot add semantically appropriate
background imperfections without affecting the foreground. Qwen can.
"""
import random
from typing import Dict, Any
from PIL import Image
from .base_tool import BaseTool

_BASE_INSTRUCTION = (
    "Make the background of this image look more natural and realistic, "
    "as if captured by a real camera in a real environment. "
    "Specifically: add subtle environmental atmosphere (slight haze or air), "
    "introduce minor focus softening at the background edges, add natural "
    "lighting variation and irregular shadows in background elements, "
    "include subtle environmental imperfections (slight dust, texture irregularity). "
    "Keep the foreground subject sharp and unchanged — only modify the background."
)

_NEGATIVE = (
    "foreground changes, subject modification, color shift, overexposed, "
    "artifacts, distorted, blurry foreground, composition change"
)


class BackgroundRealismTool(BaseTool):

    def __init__(self, model_path: str = "models/Qwen-Image-Edit-2511"):
        super().__init__(
            name="background_realism",
            description=(
                "Uses Qwen-Image-Edit-2511 to make AI's uniform perfect backgrounds "
                "look naturally imperfect. Adds environmental atmosphere, edge focus "
                "softening, lighting variation, and subtle environmental imperfections "
                "to background regions while preserving the foreground subject. "
                "Targets background perfection artifacts that PIL cannot address — requires GPU."
            )
        )
        self.model_path = model_path
        self._pipeline = None
        self._initialized = False

    def _load(self):
        if self._initialized:
            return
        from ._diffusion import load_pipeline

        self._pipeline = load_pipeline(self.name, self.model_path)
        self._initialized = True

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        cfg_scale = params.get("cfg_scale", 2.5)
        num_steps = params.get("num_steps", 28)
        environment = params.get("environment", "")

        self._load()
        import torch

        if image.mode != "RGB":
            image = image.convert("RGB")

        instruction = _BASE_INSTRUCTION
        if environment:
            instruction = f"This is an {environment} environment. " + instruction

        try:
            with torch.inference_mode():
                output = self._pipeline(
                    image=image,
                    prompt=instruction,
                    negative_prompt=_NEGATIVE,
                    true_cfg_scale=cfg_scale,
                    guidance_scale=1.0,
                    num_inference_steps=num_steps,
                    num_images_per_prompt=1,
                    generator=torch.manual_seed(random.randint(0, 2**31)),
                )
            result = output.images[0]
            if result.size != image.size:
                result = result.resize(image.size, Image.Resampling.LANCZOS)
            return result
        except Exception as e:
            print(f"background_realism failed: {e} — returning original")
            return image

    def get_guidance(self) -> str:
        return (
            "cfg_scale 2.0-2.5: subtle background imperfections (recommended). "
            "2.5-3.5: more pronounced environmental effects. "
            "environment: optionally hint scene type: 'outdoor', 'indoor office', "
            "'street', 'nature', 'urban' to guide appropriate atmosphere. "
            "Most useful when detector mentions 'perfect background', 'uniform environment', "
            "or 'unnaturally clean surroundings'."
        )

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "cfg_scale": {
                "type": "float", "min": 1.0, "max": 5.0,
                "description": "Edit strength (2.0-2.5 subtle background change, 2.5-3.5 more visible)",
                "default": 2.5, "required": False,
            },
            "num_steps": {
                "type": "int", "min": 20, "max": 50,
                "description": "Denoising steps (28 balanced)",
                "default": 28, "required": False,
            },
            "environment": {
                "type": "str",
                "description": "Scene type hint: 'outdoor', 'indoor', 'street', 'nature', 'urban', etc.",
                "default": "", "required": False,
            },
        }
