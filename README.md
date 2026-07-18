# ComfyUI-Rebels-Krea2-Outpaint

Registered outpainting for the **yijunwang2/krea2-outpaint** LoRA on **Krea 2 Turbo**, in ComfyUI.

This is a native ComfyUI port of that LoRA's diffusers pipeline. The LoRA is not a plain edit LoRA: the source image is registered into the *target* canvas grid at an explicit bounding box, so the model knows where the known pixels belong. This pack does exactly that registration and nothing else exotic — the known region is carried only by the reference, and the exact source pixels are restored at the end with a feathered seam.

## What it is under the hood

The reference-attention machinery (t=0 reference modulation, isolated-ref K/V precompute, frame axis `i+1`) is the same mechanism ostris ships in `ComfyUI-Krea2-Ostris-Edit`. The one and only difference for outpaint is the reference RoPE coordinates: instead of the reference getting its own y/x grid from `(0,0)`, its tokens are mapped into the target latent grid at the bbox with center-sampled fractional spacing. That mirrors `_pack_reference_latents` in the outpaint pipeline exactly. The reference still lives on RoPE frame `i+1` — it is **not** coplanar with the target.

## Requirements

- A ComfyUI build with native Krea 2 support (or your GGUF Krea 2 fork — see caveat below).
- Krea 2 **Turbo** diffusion model + the Krea 2 VAE + the Krea 2 Qwen3-VL CLIP.
- The LoRA weights `krea2_outpaint_rank32.safetensors` in `models/loras`.
- https://huggingface.co/yijunwang2/krea2-outpaint/blob/main/krea2_outpaint_rank32.safetensors
- No extra Python dependencies.

Install:
```
cd ComfyUI/custom_nodes
git clone https://github.com/RealRebelAI/ComfyUI-Rebels-Krea2-Outpaint
# restart ComfyUI
```

## Nodes (category: `RealRebelAI/krea2_outpaint`)

- **Rebels Krea2 Outpaint Canvas** — source + direction + target W/H → `condition`, `placed_source`, `placement` (JSON), `canvas_width`, `canvas_height`.
- **Rebels Krea2 Outpaint Encode** — clip + prompt + vae + `condition` → CONDITIONING (reference latent attached). `vlm_reference` is OFF by default, matching the LoRA's `encode_reference_in_prompt=False` contract.
- **Rebels Krea2 Outpaint Model Patch** — model + `placement` + `kv_cache` → patched MODEL. Place it **after Load LoRA**. Keep `kv_cache` ON (the LoRA uses isolated reference attention).
- **Rebels Krea2 Outpaint Composite** — generated + `placed_source` + `placement` → final image with exact known pixels restored.

## Wiring

```
Load Diffusion Model (krea2 Turbo)
    -> Load LoRA (krea2_outpaint_rank32)
    -> Rebels Krea2 Outpaint Model Patch (placement, kv_cache=True)
    -> KSampler

CLIPLoader (krea2) -> Rebels Krea2 Outpaint Encode (prompt, vae, condition) -> KSampler.positive
CLIPLoader (krea2) -> CLIP Text Encode ("")                                 -> KSampler.negative

Rebels Krea2 Outpaint Canvas:
    condition       -> Encode.condition
    placement       -> Model Patch.placement  AND  Composite.placement
    placed_source   -> Composite.placed_source
    canvas_width/height -> EmptyLatentImage

EmptyLatentImage (canvas W/H) -> KSampler.latent
KSampler -> VAE Decode -> Rebels Krea2 Outpaint Composite -> Save Image
```

## Sampler settings (Turbo)

- steps **8**, sampler **euler**, CFG **1.0** (Krea convention guidance 0.0 → CFG 1.0 in ComfyUI, which also skips the uncond pass), denoise **1.0**.
- The empty latent is the **full canvas** — pure noise, no latent masking. The reference does the anchoring.
- Prompt should describe the **complete** output image, not just the new area.

## Direction options

`extend_right`, `extend_left`, `extend_down`, `extend_up`, `extend_width_both`, `extend_height_both`. The node builds a bbox that preserves the source aspect ratio and spans the full complementary dimension (single pass). It raises with a clear message if there's no room to extend in that direction at the chosen canvas size.

## Notes / limitations

- **Single pass only** in this version — the source box spans the full width or full height. That covers extend-in-one-direction and centered-both-sides. Fully interior boxes (canvas on all four sides) need the two-pass plan; the planner is included (`plan_passes`) and the placement JSON reports `pass_count`, but chaining the two sampler passes is manual for now.
- **GGUF fork caveat:** the model patch reaches into the native Krea 2 `SingleStreamDiT` internals (`dit.first`, `dit.blocks`, `dit.pe_embedder`, `dit.patch`, block `attn`/`mod`, etc.). A GGUF loader that swaps the linear ops but keeps that module tree should work unchanged; if your fork renames or restructures those attributes, the patch is where to adjust.
- Rectangular outpainting only — this is not a mask-based inpainter.

## Credits & licenses

- Reference-attention machinery adapted from **ostris/ComfyUI-Krea2-Ostris-Edit** (MIT, © 2026 Ostris, LLC).
- Registered placement geometry and pipeline math from the **yijunwang2/krea2-outpaint** release (`pipeline.py`, `outpaint.py`, Apache-2.0; portions © 2026 Ostris, LLC and Krea AI / HuggingFace).
- This pack's node code: MIT, © 2026 RealRebelAI. See `LICENSE` and `NOTICE`.

The LoRA weights themselves are under the Krea 2 Community License; commercial misuse should be reported to Krea (opensource@krea.ai).
