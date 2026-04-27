from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


COLUMN_ALIASES = {
    "depth_um": ["depth(um)", "Depth(um)", "depth_um", "depth"],
    "distance_to_tip_um": ["distance_to_tip(um)", "Distance_To_Tip(um)", "distance_to_tip_um"],
    "inside_brain": ["inside_brain", "Inside_Brain"],
    "acronym": ["acronym", "Acronym", "structure_acronym"],
    "name": ["name", "Name", "structure_name"],
    "structure_id": ["structure_id", "id"],
    "confidence_um": [
        "distance_to_nearest_structure(um)",
        "Distance_To_Nearest_Structure(um)",
        "confidence_um",
    ],
}


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """
    Return the first candidate column present in a dataframe.
    """
    return next((col for col in candidates if col in df.columns), None)


def normalize_probe_tract_dataframe(df: pd.DataFrame, use_only_inside_brain: bool = True) -> pd.DataFrame:
    """
    Convert old and current probe tract CSV/dataframe formats to one plotting schema.
    """
    out = pd.DataFrame(index=df.index)

    for output_col, candidates in COLUMN_ALIASES.items():
        source_col = first_existing_column(df, candidates)
        if source_col is not None:
            out[output_col] = df[source_col]

    required_cols = ["depth_um", "acronym", "name", "structure_id"]
    missing = [col for col in required_cols if col not in out.columns]
    if missing:
        raise ValueError(f"Probe tract data is missing required columns: {missing}")

    out["depth_um"] = pd.to_numeric(out["depth_um"], errors="coerce")
    out["structure_id"] = pd.to_numeric(out["structure_id"], errors="coerce")

    if "distance_to_tip_um" in out.columns:
        out["distance_to_tip_um"] = pd.to_numeric(out["distance_to_tip_um"], errors="coerce")

    if "confidence_um" in out.columns:
        out["confidence_um"] = pd.to_numeric(out["confidence_um"], errors="coerce")
    else:
        out["confidence_um"] = np.nan

    if "inside_brain" in out.columns:
        out["inside_brain"] = out["inside_brain"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        out["inside_brain"] = True

    out["acronym"] = out["acronym"].astype(str)
    out["name"] = out["name"].astype(str)

    out = out.dropna(subset=["depth_um", "structure_id"]).copy()
    out["structure_id"] = out["structure_id"].astype(int)

    if use_only_inside_brain and out["inside_brain"].any():
        out = out.loc[out["inside_brain"]].copy()

    out = out.sort_values("depth_um").reset_index(drop=True)
    if out.empty:
        raise ValueError("Probe tract data has no usable rows after normalization.")

    return out


def make_sphere_offsets(radius_voxels: int) -> np.ndarray:
    """
    Build integer coordinate offsets for a sphere of radius `radius_voxels`.
    """
    grid = np.array(
        np.meshgrid(
            np.arange(-radius_voxels, radius_voxels + 1),
            np.arange(-radius_voxels, radius_voxels + 1),
            np.arange(-radius_voxels, radius_voxels + 1),
            indexing="ij",
        )
    ).reshape(3, -1).T
    distances = np.sqrt((grid ** 2).sum(axis=1))
    return grid[distances <= radius_voxels]


def estimate_confidence_from_coords(
    coords: Sequence[Sequence[int]] | pd.DataFrame | np.ndarray,
    annotation: np.ndarray,
    radius_voxels: int = 10,
    voxel_size_um: float = 10.0,
) -> np.ndarray:
    """
    Estimate distance to the nearest different atlas structure around each coordinate.
    """
    coords_array = np.asarray(coords, dtype=float)
    coords_array = np.round(coords_array).astype(int)
    shape = np.array(annotation.shape, dtype=int)
    sphere_offsets = make_sphere_offsets(radius_voxels)
    confidence_um = np.full(len(coords_array), np.nan, dtype=float)

    for idx, center in enumerate(coords_array):
        if np.any(center < 0) or np.any(center >= shape):
            continue

        current_id = annotation[center[0], center[1], center[2]]
        sphere_coords = center[None, :] + sphere_offsets
        valid = np.all((sphere_coords >= 0) & (sphere_coords < shape[None, :]), axis=1)
        sphere_coords = sphere_coords[valid]

        sphere_ids = annotation[sphere_coords[:, 0], sphere_coords[:, 1], sphere_coords[:, 2]]
        different = sphere_ids != current_id

        if not np.any(different):
            confidence_um[idx] = radius_voxels * voxel_size_um
            continue

        different_coords = sphere_coords[different]
        voxel_distances = np.sqrt(((different_coords - center[None, :]) ** 2).sum(axis=1))
        confidence_um[idx] = np.min(voxel_distances) * voxel_size_um

    return confidence_um


def rgb_triplet_to_hex(rgb_triplet: Sequence[int]) -> str:
    r, g, b = [int(value) for value in rgb_triplet]
    return f"#{r:02X}{g:02X}{b:02X}"


def get_atlas_structure_color_map(atlas) -> dict[int, str]:
    """
    Build a structure-id-to-hex-color map from BrainGlobe atlas metadata.
    """
    structures = getattr(atlas.structures, "data", atlas.structures)
    records = structures.values() if hasattr(structures, "values") else structures
    color_map = {}

    for record in records:
        structure_id = record.get("id")
        rgb_triplet = record.get("rgb_triplet")
        if structure_id is not None and rgb_triplet is not None:
            color_map[int(structure_id)] = rgb_triplet_to_hex(rgb_triplet)

    return color_map
