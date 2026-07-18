"""Canvas preparation and seam compositing for registered Krea 2 outpainting.

The RegisteredSource / PassPlan / prepare_source / plan_passes / composite block
is vendored verbatim from the yijunwang2/krea2-outpaint release's ``outpaint.py``
(pipeline/helper code, Apache-2.0). Added below it: a direction -> bbox builder
that guarantees a valid single-pass placement, ComfyUI IMAGE tensor <-> PIL
bridges, and a small JSON placement schema so the four nodes can pass the box
around without a custom ComfyUI type.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math

import numpy as np
import torch
from PIL import Image


SOURCE_MAX_EDGE = 384
SEAM_PX = 32


# ===========================================================================
# Vendored from yijunwang2/krea2-outpaint outpaint.py (Apache-2.0)
# ===========================================================================


@dataclass(frozen=True)
class RegisteredSource:
    condition: Image.Image
    placed_source: Image.Image
    canvas_size: tuple[int, int]
    bbox: tuple[int, int, int, int]
    seam_px: int = SEAM_PX

    @property
    def bbox_normalized(self) -> list[float]:
        width, height = self.canvas_size
        x0, y0, x1, y1 = self.bbox
        return [x0 / width, y0 / height, x1 / width, y1 / height]


@dataclass(frozen=True)
class PassPlan:
    axis: str
    intermediate_size: tuple[int, int] | None
    first_bbox: tuple[int, int, int, int]
    second_bbox: tuple[int, int, int, int] | None

    @property
    def pass_count(self) -> int:
        return 1 if self.intermediate_size is None else 2


def _resize_max_edge(image: Image.Image, max_edge: int) -> Image.Image:
    if max(image.size) <= max_edge:
        return image.copy()
    scale = max_edge / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def prepare_source(
    source: Image.Image,
    canvas_size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    *,
    source_max_edge: int = SOURCE_MAX_EDGE,
    seam_px: int = SEAM_PX,
) -> RegisteredSource:
    width, height = canvas_size
    x0, y0, x1, y1 = bbox
    if width < 16 or height < 16 or width % 16 or height % 16:
        raise ValueError("Canvas dimensions must be positive multiples of 16")
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Source bbox is outside the canvas: {bbox}")

    source = source.convert("RGB")
    box_width, box_height = x1 - x0, y1 - y0
    source_ratio = source.width / source.height
    box_ratio = box_width / box_height
    tolerance = max(0.025, 2.0 / min(box_width, box_height))
    if abs(box_ratio / source_ratio - 1.0) > tolerance:
        raise ValueError("Source bbox must preserve the source image aspect ratio")

    placed = source.resize((box_width, box_height), Image.Resampling.LANCZOS)
    return RegisteredSource(
        condition=_resize_max_edge(placed, source_max_edge),
        placed_source=placed,
        canvas_size=canvas_size,
        bbox=bbox,
        seam_px=seam_px,
    )


def _align_up(value: int, alignment: int = 16) -> int:
    return int(math.ceil(value / alignment) * alignment)


def plan_passes(prepared: RegisteredSource) -> PassPlan:
    width, height = prepared.canvas_size
    x0, y0, x1, y1 = prepared.bbox
    source_width, source_height = x1 - x0, y1 - y0
    if source_width == width or source_height == height:
        return PassPlan("direct", None, prepared.bbox, None)

    candidates: list[tuple[int, PassPlan]] = []
    intermediate_height = _align_up(source_height)
    if intermediate_height < height:
        intermediate_y = max(
            0,
            min(height - intermediate_height, y0 - (intermediate_height - source_height) // 2),
        )
        local_y = y0 - intermediate_y
        candidates.append(
            (
                width * intermediate_height,
                PassPlan(
                    "horizontal_first",
                    (width, intermediate_height),
                    (x0, local_y, x1, local_y + source_height),
                    (0, intermediate_y, width, intermediate_y + intermediate_height),
                ),
            )
        )

    intermediate_width = _align_up(source_width)
    if intermediate_width < width:
        intermediate_x = max(
            0,
            min(width - intermediate_width, x0 - (intermediate_width - source_width) // 2),
        )
        local_x = x0 - intermediate_x
        candidates.append(
            (
                intermediate_width * height,
                PassPlan(
                    "vertical_first",
                    (intermediate_width, height),
                    (local_x, y0, local_x + source_width, y1),
                    (intermediate_x, 0, intermediate_x + intermediate_width, height),
                ),
            )
        )

    if not candidates:
        return PassPlan("direct", None, prepared.bbox, None)
    return min(candidates, key=lambda item: (item[0], item[1].axis))[1]


def composite(generated: Image.Image, prepared: RegisteredSource) -> Image.Image:
    generated = generated.convert("RGB")
    if generated.size != prepared.canvas_size:
        raise ValueError("Generated image size does not match the canvas")

    width, height = prepared.placed_source.size
    yy, xx = np.mgrid[:height, :width]
    edge_distance = np.minimum.reduce((xx, yy, width - 1 - xx, height - 1 - yy))
    alpha = np.clip(edge_distance / max(1, prepared.seam_px), 0.0, 1.0)
    alpha_image = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")
    result = generated.copy()
    result.paste(prepared.placed_source, prepared.bbox[:2], alpha_image)
    return result


# ===========================================================================
# RealRebelAI additions
# ===========================================================================


def snap16(value: int) -> int:
    """Round to the nearest positive multiple of 16 (min 16)."""
    return max(16, int(round(value / 16.0)) * 16)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """ComfyUI IMAGE (B, H, W, C) float [0,1] -> first-frame RGB PIL."""
    if image.ndim == 4:
        image = image[0]
    arr = (image.clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return Image.fromarray(arr, mode="RGB")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """RGB PIL -> ComfyUI IMAGE (1, H, W, 3) float [0,1]."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


DIRECTIONS = [
    "extend_right",
    "extend_left",
    "extend_down",
    "extend_up",
    "extend_width_both",
    "extend_height_both",
]


def build_bbox(
    source_w: int,
    source_h: int,
    canvas_w: int,
    canvas_h: int,
    direction: str,
) -> tuple[int, int, int, int]:
    """Compute a single-pass-valid bbox that preserves the source aspect ratio
    and spans the full complementary canvas dimension.

    Horizontal extends span the full canvas HEIGHT (box width = height * aspect);
    vertical extends span the full canvas WIDTH (box height = width / aspect).
    Raises if the source cannot fit (i.e. the request would shrink, not extend).
    """
    aspect = source_w / source_h  # w / h

    if direction in ("extend_right", "extend_left", "extend_width_both"):
        box_h = canvas_h
        box_w = int(round(box_h * aspect))
        if box_w >= canvas_w:
            raise ValueError(
                f"No room for a horizontal extend: at full height {box_h} the "
                f"source needs width {box_w}, which meets or exceeds the canvas "
                f"width {canvas_w} (nothing left to outpaint). Increase target "
                f"width or pick a vertical extend."
            )
        if direction == "extend_right":
            x0 = 0
        elif direction == "extend_left":
            x0 = canvas_w - box_w
        else:  # both
            x0 = (canvas_w - box_w) // 2
        return (x0, 0, x0 + box_w, box_h)

    # vertical extends
    box_w = canvas_w
    box_h = int(round(box_w / aspect))
    if box_h >= canvas_h:
        raise ValueError(
            f"No room for a vertical extend: at full width {box_w} the source "
            f"needs height {box_h}, which meets or exceeds the canvas height "
            f"{canvas_h} (nothing left to outpaint). Increase target height or "
            f"pick a horizontal extend."
        )
    if direction == "extend_down":
        y0 = 0
    elif direction == "extend_up":
        y0 = canvas_h - box_h
    else:  # both
        y0 = (canvas_h - box_h) // 2
    return (0, y0, box_w, y0 + box_h)


def placement_to_json(prepared: RegisteredSource) -> str:
    plan = plan_passes(prepared)
    return json.dumps(
        {
            "canvas": list(prepared.canvas_size),
            "bbox": list(prepared.bbox),
            "bbox_normalized": prepared.bbox_normalized,
            "seam_px": prepared.seam_px,
            "pass_count": plan.pass_count,
            "plan_axis": plan.axis,
        }
    )


def placement_from_json(text: str) -> dict:
    data = json.loads(text)
    for key in ("canvas", "bbox", "bbox_normalized"):
        if key not in data:
            raise ValueError(f"Placement JSON missing '{key}'.")
    return data
