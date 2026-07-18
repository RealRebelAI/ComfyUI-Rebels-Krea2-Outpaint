"""RealRebelAI - Registered outpainting nodes for the yijunwang2/krea2-outpaint
LoRA on Krea 2 Turbo in ComfyUI.

Graph:

  Load Diffusion Model (krea2) -> Load LoRA (krea2_outpaint_rank32)
      -> Rebels Krea2 Outpaint Model Patch -> KSampler (cfg 1.0, 8 steps, euler)
  CLIPLoader (krea2) -> Rebels Krea2 Outpaint Encode (prompt + vae + condition)
      -> positive ; (empty negative)
  Rebels Krea2 Outpaint Canvas -> condition/placement/placed_source + W/H
  EmptyLatentImage (canvas W/H) -> KSampler -> VAE Decode
      -> Rebels Krea2 Outpaint Composite -> final image

The known region is carried ONLY by the registered reference (no latent masking);
exact source pixels are restored at the end by the composite node.
"""

import math

import torch

import comfy.conds
import node_helpers

from . import krea2_registered_core as core
from . import outpaint_helpers as oh


# ---------------------------------------------------------------------------
# 1. Canvas preparation
# ---------------------------------------------------------------------------


class RebelsKrea2OutpaintCanvas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "direction": (oh.DIRECTIONS, {"default": "extend_right"}),
                "target_width": ("INT", {"default": 1536, "min": 16, "max": 8192, "step": 16}),
                "target_height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "source_max_edge": ("INT", {"default": 384, "min": 128, "max": 768, "step": 16}),
                "seam_px": ("INT", {"default": 32, "min": 0, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("condition", "placed_source", "placement", "canvas_width", "canvas_height")
    FUNCTION = "prepare"
    CATEGORY = "RealRebelAI/krea2_outpaint"
    DESCRIPTION = (
        "Register a source image into a larger canvas at a bbox that preserves "
        "its aspect ratio and spans the full complementary dimension (single "
        "pass). Outputs the coarse condition image (for the encode node), the "
        "full-res placed source (for the composite node), a placement JSON, and "
        "the canvas size (for an EmptyLatentImage)."
    )

    def prepare(self, source, direction, target_width, target_height, source_max_edge, seam_px):
        canvas_w = oh.snap16(target_width)
        canvas_h = oh.snap16(target_height)

        src_pil = oh.tensor_to_pil(source)
        bbox = oh.build_bbox(src_pil.width, src_pil.height, canvas_w, canvas_h, direction)
        prepared = oh.prepare_source(
            src_pil,
            (canvas_w, canvas_h),
            bbox,
            source_max_edge=source_max_edge,
            seam_px=seam_px,
        )
        placement = oh.placement_to_json(prepared)

        return (
            oh.pil_to_tensor(prepared.condition),
            oh.pil_to_tensor(prepared.placed_source),
            placement,
            canvas_w,
            canvas_h,
        )


# ---------------------------------------------------------------------------
# 2. Registered encode (prompt + coarse reference latent)
# ---------------------------------------------------------------------------


def _encode_ref_latent(vae, condition_image):
    """VAE-encode the coarse condition image (snapped to /16) as a reference
    latent, matching the pipeline's 384-max-edge conditioning path."""
    samples = condition_image.movedim(-1, 1)  # (B, C, H, W)
    h, w = samples.shape[2], samples.shape[3]
    nh, nw = oh.snap16(h), oh.snap16(w)
    if (nh, nw) != (h, w):
        samples = torch.nn.functional.interpolate(
            samples, size=(nh, nw), mode="bilinear", align_corners=False
        )
    pixels = samples.movedim(1, -1)[:, :, :, :3]
    return vae.encode(pixels)


class RebelsKrea2OutpaintEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "vae": ("VAE",),
                "condition": ("IMAGE",),
            },
            "optional": {
                "vlm_reference": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "RealRebelAI/krea2_outpaint"
    DESCRIPTION = (
        "Encode the outpaint prompt and attach the coarse source as a reference "
        "latent. vlm_reference is OFF by default: the krea2-outpaint LoRA's "
        "contract is encode_reference_in_prompt=False, so the reference reaches "
        "the model only as a latent, not through the Qwen3-VL vision tower."
    )

    def encode(self, clip, prompt, vae, condition, vlm_reference=False):
        images_vl = []
        image_prompt = ""
        if vlm_reference:
            # condition is already <=384 max edge, so it fits the VL tower's
            # coarse budget; pass it straight through.
            images_vl = [condition]
            image_prompt = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"

        if images_vl:
            try:
                from comfy.text_encoders.krea2 import KREA2_TEMPLATE
                tokens = clip.tokenize(
                    image_prompt + prompt, images=images_vl, llama_template=KREA2_TEMPLATE
                )
            except Exception:
                tokens = clip.tokenize(image_prompt + prompt, images=images_vl)
        else:
            tokens = clip.tokenize(prompt)

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        ref_latent = _encode_ref_latent(vae, condition)
        conditioning = node_helpers.conditioning_set_values(
            conditioning, {"reference_latents": [ref_latent]}, append=True
        )
        return (conditioning,)


# ---------------------------------------------------------------------------
# 3. Registered model patch
# ---------------------------------------------------------------------------


class RebelsKrea2OutpaintModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "placement": ("STRING", {"forceInput": True}),
                "kv_cache": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "RealRebelAI/krea2_outpaint"
    DESCRIPTION = (
        "Patch Krea 2 so the reference latent is registered into the target grid "
        "at the placement bbox (RoPE frame i+1, target-grid y/x). kv_cache should "
        "stay ON: the krea2-outpaint LoRA uses isolated reference attention, so "
        "the refs are precomputed once at t=0 and reused every step. Place this "
        "AFTER Load LoRA and before the sampler."
    )

    def patch(self, model, placement, kv_cache=True):
        data = oh.placement_from_json(placement)
        bbox_norm = [float(v) for v in data["bbox_normalized"]]

        m = model.clone()
        base_model = m.model
        dit = m.get_model_object("diffusion_model")

        orig_extra_conds = base_model.extra_conds
        orig_extra_conds_shapes = base_model.extra_conds_shapes
        orig_forward = dit.forward

        def extra_conds(**kwargs):
            out = orig_extra_conds(**kwargs)
            ref_latents = kwargs.get("reference_latents", None)
            if ref_latents is not None:
                out["ref_latents"] = comfy.conds.CONDList(
                    [base_model.process_latent_in(lat) for lat in ref_latents]
                )
            return out

        def extra_conds_shapes(**kwargs):
            out = orig_extra_conds_shapes(**kwargs)
            ref_latents = kwargs.get("reference_latents", None)
            if ref_latents is not None:
                out["ref_latents"] = list(
                    [1, 16, sum(map(lambda a: math.prod(a.size()), ref_latents)) // 16]
                )
            return out

        state = {"last_sigma": None, "caches": {}}

        def forward(
            x,
            timesteps,
            context,
            attention_mask=None,
            transformer_options={},
            ref_latents=None,
            **kwargs,
        ):
            if ref_latents is None or len(ref_latents) == 0:
                return orig_forward(
                    x,
                    timesteps,
                    context,
                    attention_mask=attention_mask,
                    transformer_options=transformer_options,
                    **kwargs,
                )
            if not kv_cache:
                return core._forward_with_refs(
                    dit, x, timesteps, context, ref_latents, transformer_options,
                    bbox_norm=bbox_norm,
                )

            sig = float(timesteps.max())
            sample_sigmas = transformer_options.get("sample_sigmas", None)
            new_run = state["last_sigma"] is None or sig > state["last_sigma"]
            if (
                sample_sigmas is not None
                and sig == float(sample_sigmas[0])
                and sig != state["last_sigma"]
            ):
                new_run = True
            if new_run:
                state["caches"].clear()
            state["last_sigma"] = sig

            bs = x.shape[0] * (x.shape[2] if x.ndim == 5 else 1)
            key = core._ref_fingerprint(ref_latents, bs, bbox_norm)
            ref_kv = state["caches"].get(key)
            if ref_kv is None:
                ref_kv = core._precompute_ref_kv(
                    dit, x, timesteps, ref_latents, transformer_options, bbox_norm=bbox_norm
                )
                state["caches"][key] = ref_kv
            return core._forward_with_cached_refs(
                dit, x, timesteps, context, ref_kv, transformer_options
            )

        m.add_object_patch("extra_conds", extra_conds)
        m.add_object_patch("extra_conds_shapes", extra_conds_shapes)
        m.add_object_patch("diffusion_model.forward", forward)
        return (m,)


# ---------------------------------------------------------------------------
# 4. Seam composite
# ---------------------------------------------------------------------------


class RebelsKrea2OutpaintComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated": ("IMAGE",),
                "placed_source": ("IMAGE",),
                "placement": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"
    CATEGORY = "RealRebelAI/krea2_outpaint"
    DESCRIPTION = (
        "Restore the exact known source pixels into the generated canvas with an "
        "inward feather (seam_px from the placement), hiding decode differences "
        "at the boundary."
    )

    def composite(self, generated, placed_source, placement):
        data = oh.placement_from_json(placement)
        canvas_size = (int(data["canvas"][0]), int(data["canvas"][1]))
        bbox = tuple(int(v) for v in data["bbox"])
        seam_px = int(data.get("seam_px", oh.SEAM_PX))

        gen_pil = oh.tensor_to_pil(generated)
        placed_pil = oh.tensor_to_pil(placed_source)

        prepared = oh.RegisteredSource(
            condition=placed_pil,  # unused by composite
            placed_source=placed_pil,
            canvas_size=canvas_size,
            bbox=bbox,
            seam_px=seam_px,
        )
        result = oh.composite(gen_pil, prepared)
        return (oh.pil_to_tensor(result),)


NODE_CLASS_MAPPINGS = {
    "RebelsKrea2OutpaintCanvas": RebelsKrea2OutpaintCanvas,
    "RebelsKrea2OutpaintEncode": RebelsKrea2OutpaintEncode,
    "RebelsKrea2OutpaintModelPatch": RebelsKrea2OutpaintModelPatch,
    "RebelsKrea2OutpaintComposite": RebelsKrea2OutpaintComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RebelsKrea2OutpaintCanvas": "Rebels Krea2 Outpaint Canvas",
    "RebelsKrea2OutpaintEncode": "Rebels Krea2 Outpaint Encode",
    "RebelsKrea2OutpaintModelPatch": "Rebels Krea2 Outpaint Model Patch",
    "RebelsKrea2OutpaintComposite": "Rebels Krea2 Outpaint Composite",
}
