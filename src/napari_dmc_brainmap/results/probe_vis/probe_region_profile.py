from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np
import pandas as pd

from napari_dmc_brainmap.results.probe_vis.probe_confidence import normalize_probe_tract_dataframe


@dataclass(frozen=True)
class ProbeRegionProfileConfig:
    figsize: tuple[float, float] = (2.2, 8.0)
    dpi: int = 400
    background_color: str = "white"
    background_bar_color: str = "#E6E6E6"
    confidence_max_um: float = 100.0
    min_label_height_um: float = 40.0
    label_side: str = "right"
    show_region_labels: bool = True
    show_confidence_axis: bool = True
    save_png: bool = True
    save_svg: bool = True


def confidence_um_to_x(confidence_values: np.ndarray, confidence_max_um: float = 100.0) -> np.ndarray:
    confidence_values = np.asarray(confidence_values, dtype=float)
    confidence_x = confidence_values / confidence_max_um
    return np.clip(confidence_x, 0, 1)


def build_segments(df: pd.DataFrame) -> list[dict]:
    depths = df["depth_um"].to_numpy(dtype=float)
    acronyms = df["acronym"].astype(str).to_numpy()
    names = df["name"].astype(str).to_numpy()
    structure_ids = df["structure_id"].to_numpy(dtype=int)
    confidence = df["confidence_um"].to_numpy(dtype=float)

    segments = []
    start_idx = 0

    for idx in range(1, len(df)):
        if acronyms[idx] != acronyms[idx - 1]:
            segments.append(
                {
                    "depths": depths[start_idx:idx],
                    "acronym": acronyms[start_idx],
                    "name": names[start_idx],
                    "structure_id": int(structure_ids[start_idx]),
                    "confidence": confidence[start_idx:idx],
                }
            )
            start_idx = idx

    segments.append(
        {
            "depths": depths[start_idx:],
            "acronym": acronyms[start_idx],
            "name": names[start_idx],
            "structure_id": int(structure_ids[start_idx]),
            "confidence": confidence[start_idx:],
        }
    )

    return segments


def segment_bounds_from_depths(depths: np.ndarray) -> np.ndarray:
    depths = np.asarray(depths, dtype=float)

    if len(depths) == 1:
        depth = depths[0]
        return np.array([depth - 10, depth + 10], dtype=float)

    diffs = np.diff(depths)
    nonzero_diffs = diffs[diffs != 0]
    fallback = np.nanmedian(nonzero_diffs) if len(nonzero_diffs) else 20.0
    diffs = np.where(diffs == 0, fallback, diffs)

    bounds = np.empty(len(depths) + 1, dtype=float)
    bounds[1:-1] = (depths[:-1] + depths[1:]) / 2
    bounds[0] = depths[0] - diffs[0] / 2
    bounds[-1] = depths[-1] + diffs[-1] / 2
    return bounds


def _draw_confidence_region(
    ax,
    y_bounds: np.ndarray,
    confidence_values: np.ndarray,
    color: str,
    config: ProbeRegionProfileConfig,
) -> None:
    ax.fill_betweenx(y_bounds, 0.0, 1.0, color=config.background_bar_color, linewidth=0, zorder=1)

    if confidence_values is None or len(confidence_values) == 0:
        return

    confidence_values = np.asarray(confidence_values, dtype=float)
    if np.all(np.isnan(confidence_values)):
        return

    if len(confidence_values) == 1:
        confidence_x = np.array([confidence_values[0], confidence_values[0]], dtype=float)
    else:
        confidence_x = np.concatenate([confidence_values, [confidence_values[-1]]])

    confidence_x = confidence_um_to_x(confidence_x, config.confidence_max_um)
    confidence_x = np.nan_to_num(confidence_x, nan=0.0)
    ax.fill_betweenx(y_bounds, 0.0, confidence_x, color=color, linewidth=0, zorder=2)


def save_probe_region_confidence_profile(
    probe_tract: pd.DataFrame,
    atlas_color_map: dict[int, str],
    animal_id: str,
    probe_name: str,
    output_dir: Path,
    global_ymax: Optional[float] = None,
    config: ProbeRegionProfileConfig = ProbeRegionProfileConfig(),
) -> tuple[Optional[Path], Optional[Path]]:
    """
    Save a compact probe region profile where confidence is encoded as bar width.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = normalize_probe_tract_dataframe(probe_tract)
    segments = build_segments(df)

    fig, ax = plt.subplots(figsize=config.figsize)
    fig.patch.set_facecolor(config.background_color)
    ax.set_facecolor(config.background_color)
    text_transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    for segment in segments:
        color = atlas_color_map.get(int(segment["structure_id"]), "#B0B0B0")
        y_bounds = segment_bounds_from_depths(segment["depths"])

        _draw_confidence_region(ax, y_bounds, segment["confidence"], color, config)

        height = y_bounds[-1] - y_bounds[0]
        y_center = 0.5 * (y_bounds[0] + y_bounds[-1])
        if config.show_region_labels and height >= config.min_label_height_um:
            if config.label_side.lower() == "left":
                ax.text(-0.06, y_center, segment["acronym"], transform=text_transform,
                        va="center", ha="right", fontsize=10, color="black")
            else:
                ax.text(1.06, y_center, segment["acronym"], transform=text_transform,
                        va="center", ha="left", fontsize=10, color="black")

    ax.set_title(f"{animal_id} - {probe_name}", fontsize=12)
    ax.set_ylabel("Depth from dura (um)", fontsize=10)

    if config.show_confidence_axis:
        ax.set_xlabel("Confidence", fontsize=6)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "50", "100"], fontsize=6)
        ax.tick_params(axis="x", labelsize=6, length=2)
    else:
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlim(-0.15, 1.15)

    ymin = max(0.0, float(np.nanmin(df["depth_um"])))
    ymax = global_ymax if global_ymax is not None else float(np.nanmax(df["depth_um"])) * 1.02
    if ymax <= ymin:
        ymax = ymin + 20.0
    ax.set_ylim(ymax, ymin)

    if config.label_side.lower() == "left":
        plt.subplots_adjust(left=0.42, right=0.88, top=0.95, bottom=0.08)
    else:
        plt.subplots_adjust(left=0.18, right=0.60, top=0.95, bottom=0.08)

    png_path = output_dir / f"{probe_name}_region_confidence_profile.png"
    svg_path = output_dir / f"{probe_name}_region_confidence_profile.svg"

    saved_png = None
    saved_svg = None
    if config.save_png:
        fig.savefig(png_path, dpi=config.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        saved_png = png_path
    if config.save_svg:
        fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        saved_svg = svg_path

    plt.close(fig)
    return saved_png, saved_svg
