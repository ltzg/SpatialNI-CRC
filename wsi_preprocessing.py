from __future__ import annotations

"""
Step 01 - WSI preprocessing.

Use this module first to open whole-slide images and generate the notebook-style
Otsu-masked 512 x 512 patch grid used by the SpatialNI workflow.
"""


from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TileSpec:
    slide_id: str
    x: int
    y: int
    width: int
    height: int
    tissue_fraction: float


@dataclass(frozen=True)
class WSIMetadata:
    file_path: str
    slide_dimensions: tuple[int, int] | None
    level_count: int | None
    objective_power: float | None
    mpp: tuple[float, float] | None
    vendor: str | None


def _require_tiatoolbox():
    try:
        import tiatoolbox.wsicore.wsireader as wsireader_module
        from tiatoolbox.tools import patchextraction
        from tiatoolbox.wsicore.wsireader import WSIReader
    except ImportError as exc:
        raise ImportError(
            "TIAToolbox is required for the notebook-style WSI preprocessing helpers."
        ) from exc
    wsireader_module.is_ngff = lambda input_path: False
    return WSIReader, patchextraction


def open_wsi(wsi_path: str):
    """Open a WSI with TIAToolbox, following the notebook workflow."""
    WSIReader, _ = _require_tiatoolbox()
    return WSIReader.open(wsi_path)


def read_wsi_metadata(wsi_path: str) -> WSIMetadata:
    reader = open_wsi(wsi_path)
    info = reader.info.as_dict()
    return WSIMetadata(
        file_path=str(info.get("file_path", wsi_path)),
        slide_dimensions=tuple(info["slide_dimensions"]) if info.get("slide_dimensions") is not None else None,
        level_count=int(info["level_count"]) if info.get("level_count") is not None else None,
        objective_power=float(info["objective_power"]) if info.get("objective_power") is not None else None,
        mpp=tuple(info["mpp"]) if info.get("mpp") is not None else None,
        vendor=str(info["vendor"]) if info.get("vendor") is not None else None,
    )


def read_slide_thumbnail(
    wsi_path: str,
    resolution: float = 1.25,
    units: str = "power",
) -> np.ndarray:
    """
    Read a thumbnail exactly in the style used in the reference notebook:
    `reader.slide_thumbnail(resolution=1.25, units="power")`.
    """
    reader = open_wsi(wsi_path)
    return np.asarray(reader.slide_thumbnail(resolution=resolution, units=units))


def build_tiatoolbox_patch_extractor(
    wsi_path: str,
    patch_size: tuple[int, int] = (512, 512),
    stride: tuple[int, int] = (512, 512),
    resolution: float = 20.0,
    units: str = "power",
    input_mask: str | None = None,
):
    """
    Build a sliding-window patch extractor using TIAToolbox.

    This mirrors the notebook pattern:
    - `WSIReader.open(...)`
    - `patchextraction.get_patch_extractor(...)`
    """
    _, patchextraction = _require_tiatoolbox()
    reader = open_wsi(wsi_path)
    kwargs = {
        "input_img": reader,
        "method_name": "slidingwindow",
        "patch_size": patch_size,
        "stride": stride,
        "resolution": resolution,
        "units": units,
    }
    if input_mask is not None:
        kwargs["input_mask"] = input_mask
    return patchextraction.get_patch_extractor(**kwargs)


def notebook_style_patch_locations(
    wsi_path: str | Path,
    patch_size: tuple[int, int] = (512, 512),
    stride: tuple[int, int] = (512, 512),
    resolution: float = 20.0,
    units: str = "power",
    input_mask: str = "otsu",
) -> pd.DataFrame:
    """
    Return the WSI patch coordinates used by the original notebook workflow.

    The manuscript TCGA workflow used TIAToolbox sliding-window extraction with:
    512 x 512 patches, 512 stride, 20x objective power, and an Otsu tissue mask.
    The returned `x` and `y` coordinates are in the requested resolution space.
    """
    patch_extractor = build_tiatoolbox_patch_extractor(
        str(wsi_path),
        patch_size=patch_size,
        stride=stride,
        resolution=resolution,
        units=units,
        input_mask=input_mask,
    )
    locations = extractor_locations_to_dataframe(patch_extractor)
    if "x" not in locations.columns or "y" not in locations.columns:
        raise KeyError("TIAToolbox locations must contain `x` and `y` columns.")
    locations = locations.copy()
    locations["x"] = locations["x"].astype(int)
    locations["y"] = locations["y"].astype(int)
    locations["patch_width"] = int(patch_size[0])
    locations["patch_height"] = int(patch_size[1])
    locations["resolution"] = float(resolution)
    locations["units"] = units
    locations["input_mask"] = input_mask
    locations["slide_id"] = Path(wsi_path).stem
    return locations.reset_index(drop=True)


def extractor_locations_to_dataframe(patch_extractor) -> pd.DataFrame:
    """
    Return TIAToolbox patch locations as a tidy dataframe.

    The notebook uses `patch_extractor.locations_df` and `patch_extractor.coordinate_list`.
    """
    if hasattr(patch_extractor, "locations_df"):
        return patch_extractor.locations_df.copy()
    if hasattr(patch_extractor, "coordinate_list"):
        coords = np.asarray(patch_extractor.coordinate_list)
        if coords.ndim == 2 and coords.shape[1] >= 4:
            return pd.DataFrame(coords[:, :4], columns=["x_start", "y_start", "x_end", "y_end"])
    raise AttributeError("Patch extractor does not expose `locations_df` or `coordinate_list`.")


def read_patches_from_locations(
    wsi_path: str,
    locations_df: pd.DataFrame,
    patch_size: tuple[int, int] = (512, 512),
    resolution: float = 20.0,
    units: str = "power",
    method: str = "read_bounds",
) -> list[np.ndarray]:
    """
    Read patch arrays from TIAToolbox extractor coordinates.

    Supported methods:
    - `read_bounds`: reads TIAToolbox extractor coordinates without changing coordinate space
    - `read_rect`: reads a rectangle from a location; useful only when coordinates are already in
      the requested read resolution space
    """
    reader = open_wsi(wsi_path)
    patches: list[np.ndarray] = []

    x_col = "x" if "x" in locations_df.columns else "x_start"
    y_col = "y" if "y" in locations_df.columns else "y_start"

    for _, row in locations_df.iterrows():
        x = int(row[x_col])
        y = int(row[y_col])
        if method == "read_rect":
            patch = reader.read_rect(
                location=(x, y),
                size=patch_size,
                resolution=resolution,
                units=units,
            )
        elif method == "read_bounds":
            patch = reader.read_bounds(
                bounds=[x, y, x + patch_size[0], y + patch_size[1]],
                resolution=resolution,
                units=units,
                coord_space="resolution",
            )
        else:
            raise ValueError("method must be either `read_rect` or `read_bounds`.")
        patches.append(np.asarray(patch))
    return patches


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image in [0, 255] to HSV in [0, 1]."""
    rgb = np.asarray(rgb, dtype=np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    deltac = maxc - minc

    hue = np.zeros_like(maxc)
    sat = np.zeros_like(maxc)
    val = maxc

    nonzero = deltac > 0
    sat[maxc > 0] = deltac[maxc > 0] / maxc[maxc > 0]

    red = nonzero & (maxc == r)
    green = nonzero & (maxc == g)
    blue = nonzero & (maxc == b)

    hue[red] = ((g[red] - b[red]) / deltac[red]) % 6.0
    hue[green] = ((b[green] - r[green]) / deltac[green]) + 2.0
    hue[blue] = ((r[blue] - g[blue]) / deltac[blue]) + 4.0
    hue /= 6.0

    return np.stack([hue, sat, val], axis=-1)


def estimate_tissue_mask(
    thumbnail_rgb: np.ndarray,
    white_value_threshold: float = 0.85,
    low_saturation_threshold: float = 0.05,
) -> np.ndarray:
    """
    Build a simple tissue mask from a low-resolution RGB thumbnail.

    Heuristic:
    - background is usually bright and weakly saturated
    - tissue tends to have lower value and stronger saturation
    """
    hsv = rgb_to_hsv(thumbnail_rgb)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    not_white = value < white_value_threshold
    not_blank = saturation > low_saturation_threshold
    return (not_white | not_blank).astype(bool)


def generate_patch_grid(
    width: int,
    height: int,
    patch_size: int = 512,
    stride: int = 512,
) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for y in range(0, max(height - patch_size + 1, 1), stride):
        for x in range(0, max(width - patch_size + 1, 1), stride):
            coords.append((x, y))
    return coords


def _tile_fraction_from_mask(
    mask: np.ndarray,
    tile_x: int,
    tile_y: int,
    tile_size: int,
    full_width: int,
    full_height: int,
) -> float:
    mask_h, mask_w = mask.shape
    x0 = int(np.floor(tile_x / full_width * mask_w))
    y0 = int(np.floor(tile_y / full_height * mask_h))
    x1 = int(np.ceil((tile_x + tile_size) / full_width * mask_w))
    y1 = int(np.ceil((tile_y + tile_size) / full_height * mask_h))
    x0 = max(0, min(mask_w, x0))
    y0 = max(0, min(mask_h, y0))
    x1 = max(x0 + 1, min(mask_w, x1))
    y1 = max(y0 + 1, min(mask_h, y1))
    view = mask[y0:y1, x0:x1]
    return float(view.mean()) if view.size else 0.0


def select_tissue_tiles(
    slide_id: str,
    slide_width: int,
    slide_height: int,
    tissue_mask: np.ndarray,
    patch_size: int = 512,
    stride: int = 512,
    min_tissue_fraction: float = 0.25,
) -> list[TileSpec]:
    tiles: list[TileSpec] = []
    for x, y in generate_patch_grid(slide_width, slide_height, patch_size, stride):
        frac = _tile_fraction_from_mask(
            mask=tissue_mask,
            tile_x=x,
            tile_y=y,
            tile_size=patch_size,
            full_width=slide_width,
            full_height=slide_height,
        )
        if frac >= min_tissue_fraction:
            tiles.append(
                TileSpec(
                    slide_id=slide_id,
                    x=x,
                    y=y,
                    width=patch_size,
                    height=patch_size,
                    tissue_fraction=frac,
                )
            )
    return tiles


def crop_tiles_from_array(
    slide_rgb: np.ndarray,
    tiles: Iterable[TileSpec],
) -> list[np.ndarray]:
    patches: list[np.ndarray] = []
    for tile in tiles:
        patch = slide_rgb[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
        if patch.shape[:2] == (tile.height, tile.width):
            patches.append(patch.copy())
    return patches


def summarize_tile_selection(tiles: Iterable[TileSpec]) -> dict[str, float]:
    tiles = list(tiles)
    if not tiles:
        return {"n_tiles": 0.0, "mean_tissue_fraction": 0.0, "median_tissue_fraction": 0.0}
    fractions = np.array([tile.tissue_fraction for tile in tiles], dtype=float)
    return {
        "n_tiles": float(len(tiles)),
        "mean_tissue_fraction": float(np.mean(fractions)),
        "median_tissue_fraction": float(np.median(fractions)),
    }
