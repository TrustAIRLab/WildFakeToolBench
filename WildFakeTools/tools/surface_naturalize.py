"""
Surface Naturalize Tool — reduce AI's characteristic plastic/glossy look.

Source: Reddit 1rul211 Flux.2 Klein LoRA post: "Instead of smoothing out features,
the model strives to preserve and even restore the micro-details present in real skin,
hair, and fabric." + Reddit 1blph1s: "fix the jarring errors."

AI diffusion models systematically over-smooth skin, hair, and fabric surfaces,
producing a 'plastic' or 'oily' look that VLM detectors can identify. PIL-based tools
can only add generic noise — they cannot reconstruct realistic surface microstructure.
Qwen-Image-Edit-2511 can semantically re-texture surfaces toward photographic realism.

This tool targets the surface plasticity artifact specifically, using a fixed focused
instruction rather than the general-purpose iterative_denoise_pass interface.
"""
import random
from typing import Dict, Any
from PIL import Image
from .base_tool import BaseTool

_BASE_INSTRUCTION = (
    "Make the surface textures in this image look more natural and photographic. "
    "Specifically: reduce overly smooth or plastic-looking skin, hair, and fabric surfaces; "
    "add subtle realistic micro-texture variation (pores, strand detail, weave pattern); "
    "reduce uniform glossy highlights and replace with matte, diffuse surface response. "
    "Preserve composition, colors, and overall structure — only change surface texture quality."
)

_NEGATIVE = (
    "blurry, distorted, artifacts, color shift, composition change, unrealistic, "
    "over-processed, caricature, cartoon, painting, sketch"
)


class SurfaceNaturalizeTool(BaseTool):

    def __init__(self, model_path: str = "models/Qwen-Image-Edit-2511"):
        super().__init__(
            name="surface_naturalize",
            description=(
                "Uses Qwen-Image-Edit-2511 to reduce AI's characteristic plastic/glossy "
                "surface look. Semantically re-textures skin, hair, and fabric surfaces "
                "toward photographic micro-detail (pores, strand variation, weave texture) "
                "without changing composition or colors. Targets the over-smooth surface "
                "artifact that PIL-based tools cannot address — requires GPU."
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
        extra_instruction = params.get("extra_instruction", "")

        self._load()
        import torch

        if image.mode != "RGB":
            image = image.convert("RGB")

        instruction = _BASE_INSTRUCTION
        if extra_instruction:
            instruction = extra_instruction + " " + _BASE_INSTRUCTION

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
            print(f"surface_naturalize failed: {e} — returning original")
            return image

    def get_guidance(self) -> str:
        return (
            "cfg_scale 2.0-2.5: subtle texture refinement (recommended starting point). "
            "2.5-3.5: moderate re-texturing. >3.5: significant surface change. "
            "extra_instruction: optional prefix to target specific area "
            "(e.g. 'Focus on the face: '). "
            "Most useful when detector mentions 'smooth', 'plastic', 'oily', 'artificial skin'."
        )

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "cfg_scale": {
                "type": "float", "min": 1.0, "max": 5.0,
                "description": "Edit strength: 2.0-2.5 subtle, 2.5-3.5 moderate, >3.5 strong",
                "default": 2.5, "required": False,
            },
            "num_steps": {
                "type": "int", "min": 20, "max": 50,
                "description": "Denoising steps (28 balanced, 40 best quality)",
                "default": 28, "required": False,
            },
            "extra_instruction": {
                "type": "str",
                "description": "Optional prefix to focus on specific region (e.g. 'Focus on the face: ')",
                "default": "", "required": False,
            },
        }
