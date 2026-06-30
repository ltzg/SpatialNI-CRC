from __future__ import annotations

"""
Step 03 - SpatialNI feature engineering.

Run this after patch classification. This module converts patch coordinates and
tissue labels into 3 x 3 directional neighbourhood counts, normalized NI values,
composite indices, and WSI patch class maps.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TISSUE_ORDER = ["BACK", "NORM", "DEB", "TUM", "ADI", "MUC", "MUS", "STR", "LYM"]
NON_BACK_TISSUES = [label for label in TISSUE_ORDER if label != "BACK"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(TISSUE_ORDER)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
TISSUE_COLORS = {
    "NORM": "#4DBBD5",
    "DEB": "#8491B4",
    "TUM": "#E64B35",
    "ADI": "#F39B7F",
    "MUC": "#91D1C2",
    "MUS": "#7E6148",
    "STR": "#3C5488",
    "LYM": "#00A087",
    "BACK": "#BDBDBD",
}
PATCH_NAME_RE = re.compile(r"_(?P<class_id>\d+)_(?P<x>\d+)_(?P<y>\d+)_\.(?:jpg|jpeg|png)$", re.IGNORECASE)
NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


@dataclass(frozen=True)
class CompositeDefinition:
    label: str
    formula: str


COMPOSITE_DEFINITIONS = [
    CompositeDefinition(
        label="Tumor-stromal interface score",
        formula="TUM_to_STR_NI + STR_to_TUM_NI",
    ),
    CompositeDefinition(
        label="Bidirectional tumor-stromal interface score",
        formula="harmonic_mean(TUM_to_STR_NI, STR_to_TUM_NI)",
    ),
    CompositeDefinition(
        label="Tumor-muscular interface score",
        formula="TUM_to_MUS_NI + MUS_to_TUM_NI",
    ),
    CompositeDefinition(
        label="Bidirectional tumor-muscular interface score",
        formula="harmonic_mean(TUM_to_MUS_NI, MUS_to_TUM_NI)",
    ),
    CompositeDefinition(
        label="Mesenchymal interface burden score",
        formula="TUM_to_STR_NI + STR_to_TUM_NI + TUM_to_MUS_NI + MUS_to_TUM_NI",
    ),
    CompositeDefinition(
        label="Transmuscular extramural extension score",
        formula="TUM_to_MUS_NI + MUS_to_TUM_NI + TUM_to_ADI_NI + ADI_to_TUM_NI",
    ),
    CompositeDefinition(
        label="Invasive-front composite score",
        formula="TUM_to_STR_NI + STR_to_TUM_NI + TUM_to_MUS_NI + MUS_to_TUM_NI + TUM_to_ADI_NI + ADI_to_TUM_NI",
    ),
]


def infer_patch_step(coords: pd.Series) -> int:
    unique_sorted = np.sort(pd.to_numeric(coords, errors="coerce").dropna().unique())
    if len(unique_sorted) < 2:
        return 1
    diffs = np.diff(unique_sorted)
    diffs = diffs[diffs > 0]
    return int(diffs.min()) if len(diffs) else 1


def harmonic_mean_series(a: pd.Series, b: pd.Series) -> pd.Series:
    denom = a + b
    out = pd.Series(0.0, index=a.index, dtype=float)
    valid = denom != 0
    out.loc[valid] = 2.0 * a.loc[valid] * b.loc[valid] / denom.loc[valid]
    return out


def _compute_single_group(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
) -> dict[str, float]:
    step_x = infer_patch_step(frame[x_col])
    step_y = infer_patch_step(frame[y_col])
    step = max(1, min(step_x, step_y))

    patch_map = {
        (int(row[x_col]), int(row[y_col])): str(row[label_col])
        for _, row in frame[[x_col, y_col, label_col]].dropna().iterrows()
    }
    base_counts = {label: 0 for label in TISSUE_ORDER}
    neighbor_counts = {
        src: {dst: 0 for dst in TISSUE_ORDER}
        for src in TISSUE_ORDER
    }

    for (x, y), src in patch_map.items():
        if src not in base_counts:
            continue
        base_counts[src] += 1
        neighbors_found = set()
        for dx, dy in NEIGHBOR_OFFSETS:
            coord = (x + dx * step, y + dy * step)
            dst = patch_map.get(coord)
            if dst is not None:
                neighbors_found.add(dst)
        for dst in neighbors_found:
            if dst in neighbor_counts[src]:
                neighbor_counts[src][dst] += 1

    row: dict[str, float] = {}
    for src in TISSUE_ORDER:
        row[f"Count_{src}"] = float(base_counts[src])
    for src in TISSUE_ORDER:
        total = base_counts[src]
        for dst in TISSUE_ORDER:
            count = neighbor_counts[src][dst]
            ratio = round(count / total, 4) if total > 0 else 0.0
            row[f"{src}_has_{dst}_count"] = float(count)
            row[f"{src}_has_{dst}_ratio"] = float(ratio)
            row[f"{src}_to_{dst}_count"] = float(count)
            row[f"{src}_to_{dst}_NI"] = float(ratio)
    return row


def build_patient_spatialni(
    patch_df: pd.DataFrame,
    patient_col: str = "patient_id",
    x_col: str = "x",
    y_col: str = "y",
    label_col: str = "label",
) -> pd.DataFrame:
    required = {patient_col, x_col, y_col, label_col}
    missing = required.difference(patch_df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for patient_id, group in patch_df.groupby(patient_col, sort=False):
        stats = _compute_single_group(group, x_col=x_col, y_col=y_col, label_col=label_col)
        stats[patient_col] = patient_id
        rows.append(stats)

    result = pd.DataFrame(rows)
    return add_pathology_composite_scores(result)


def build_patient_spatialni_from_predictions(
    predictions: pd.DataFrame,
    patient_id: str | None = None,
    patient_col: str = "Patient_ID",
    x_col: str = "x",
    y_col: str = "y",
    label_col: str = "predicted_class_label",
) -> pd.DataFrame:
    """
    Build patient-level SpatialNI features from WSI patch predictions.

    This is the direct downstream step for `classify_wsi_with_tiatoolbox`.
    The neighbor definition is the same 8-neighbor tile adjacency used by
    `build_patient_spatialni`.
    """
    required = {x_col, y_col, label_col}
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"Missing required prediction columns: {sorted(missing)}")

    patch_df = predictions.copy()
    if patient_col not in patch_df.columns:
        if patient_id is None:
            patient_id = str(patch_df.get("slide_id", pd.Series(["sample"])).iloc[0])
        patch_df[patient_col] = patient_id
    if label_col != "label":
        patch_df["label"] = patch_df[label_col]
        label_col = "label"

    return build_patient_spatialni(
        patch_df,
        patient_col=patient_col,
        x_col=x_col,
        y_col=y_col,
        label_col=label_col,
    )


def parse_classified_patch_name(path_or_name: str | Path) -> dict[str, int | str]:
    name = Path(path_or_name).name
    match = PATCH_NAME_RE.search(name)
    if match is None:
        raise ValueError(f"Unsupported classified patch name: {name}")
    class_id = int(match.group("class_id"))
    if class_id not in ID_TO_LABEL:
        raise ValueError(f"Unsupported tissue class id {class_id}: {name}")
    return {
        "x": int(match.group("x")),
        "y": int(match.group("y")),
        "label": ID_TO_LABEL[class_id],
    }


def patch_filenames_to_dataframe(
    filenames: list[str] | pd.Series,
    patient_id: str,
    patient_col: str = "Patient_ID",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for filename in filenames:
        try:
            parsed = parse_classified_patch_name(filename)
        except ValueError:
            continue
        parsed[patient_col] = patient_id
        rows.append(parsed)
    return pd.DataFrame(rows)


def build_patient_spatialni_from_filenames(
    patient_files: dict[str, list[str]],
    patient_col: str = "Patient_ID",
) -> pd.DataFrame:
    patch_frames = [
        patch_filenames_to_dataframe(filenames, patient_id=patient_id, patient_col=patient_col)
        for patient_id, filenames in patient_files.items()
    ]
    patch_frames = [frame for frame in patch_frames if not frame.empty]
    if not patch_frames:
        return pd.DataFrame()
    patch_df = pd.concat(patch_frames, ignore_index=True)
    return build_patient_spatialni(
        patch_df,
        patient_col=patient_col,
        x_col="x",
        y_col="y",
        label_col="label",
    )


def read_directory_tree_txt(txt_path: str | Path) -> dict[str, list[str]]:
    patient_files: dict[str, list[str]] = {}
    current_patient: str | None = None
    with Path(txt_path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            stripped = line.strip()
            indent = line[: len(line) - len(line.lstrip(" \t"))]
            level = indent.count("\t") if "\t" in indent else len(indent) // 4
            if level == 1:
                current_patient = stripped
                patient_files.setdefault(current_patient, [])
            elif level >= 2 and current_patient and stripped.lower().endswith((".jpg", ".jpeg", ".png")):
                patient_files[current_patient].append(stripped)
    return patient_files


def spatial_table_for_main_analysis(patient_df: pd.DataFrame, patient_col: str = "Patient_ID") -> pd.DataFrame:
    keep_cols = [patient_col]
    for label in TISSUE_ORDER:
        keep_cols.append(f"Count_{label}")
    for src in TISSUE_ORDER:
        for dst in TISSUE_ORDER:
            keep_cols.extend([f"{src}_has_{dst}_count", f"{src}_has_{dst}_ratio"])
    return patient_df[[col for col in keep_cols if col in patient_df.columns]].copy()


def add_pathology_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    tum_str = pd.to_numeric(work.get("TUM_to_STR_NI"), errors="coerce").fillna(0.0)
    str_tum = pd.to_numeric(work.get("STR_to_TUM_NI"), errors="coerce").fillna(0.0)
    tum_mus = pd.to_numeric(work.get("TUM_to_MUS_NI"), errors="coerce").fillna(0.0)
    mus_tum = pd.to_numeric(work.get("MUS_to_TUM_NI"), errors="coerce").fillna(0.0)
    tum_adi = pd.to_numeric(work.get("TUM_to_ADI_NI"), errors="coerce").fillna(0.0)
    adi_tum = pd.to_numeric(work.get("ADI_to_TUM_NI"), errors="coerce").fillna(0.0)

    work["Tumor-stromal interface score"] = tum_str + str_tum
    work["Bidirectional tumor-stromal interface score"] = harmonic_mean_series(tum_str, str_tum)
    work["Tumor-muscular interface score"] = tum_mus + mus_tum
    work["Bidirectional tumor-muscular interface score"] = harmonic_mean_series(tum_mus, mus_tum)
    work["Mesenchymal interface burden score"] = tum_str + str_tum + tum_mus + mus_tum
    work["Transmuscular extramural extension score"] = tum_mus + mus_tum + tum_adi + adi_tum
    work["Invasive-front composite score"] = tum_str + str_tum + tum_mus + mus_tum + tum_adi + adi_tum
    return work


def long_format_neighborhoods(
    patient_df: pd.DataFrame,
    patient_col: str = "patient_id",
) -> pd.DataFrame:
    value_cols = [col for col in patient_df.columns if col.endswith("_NI")]
    long_df = patient_df.melt(
        id_vars=[patient_col],
        value_vars=value_cols,
        var_name="feature",
        value_name="value",
    )
    parts = long_df["feature"].str.extract(r"^(?P<source>[A-Z]+)_to_(?P<target>[A-Z]+)_NI$")
    long_df["source_tissue"] = parts["source"]
    long_df["target_tissue"] = parts["target"]
    return long_df


def plot_wsi_patch_class_map(
    predictions: pd.DataFrame,
    output_png: str | Path,
    wsi_path: str | Path | None = None,
    patch_size: int = 512,
    patch_resolution: float = 20.0,
    overview_resolution: float = 1.25,
    units: str = "power",
    x_col: str = "x",
    y_col: str = "y",
    label_col: str = "predicted_class_label",
    title: str | None = None,
    alpha: float = 0.55,
    dpi: int = 600,
) -> Path:
    """
    Draw a WSI patch class map like the notebook/website overview figure.

    If `wsi_path` is supplied, the patch map is overlaid on a low-resolution
    WSI thumbnail. Otherwise, only the colored patch grid is drawn.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    required = {x_col, y_col, label_col}
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"Missing required prediction columns: {sorted(missing)}")

    frame = predictions[[x_col, y_col, label_col]].dropna().copy()
    frame[x_col] = pd.to_numeric(frame[x_col], errors="coerce")
    frame[y_col] = pd.to_numeric(frame[y_col], errors="coerce")
    frame = frame.dropna(subset=[x_col, y_col])
    if frame.empty:
        raise ValueError("No valid patch predictions to plot.")

    background = None
    if wsi_path is not None:
        try:
            from wsi_preprocessing import read_slide_thumbnail
        except ImportError:
            from .wsi_preprocessing import read_slide_thumbnail
        background = read_slide_thumbnail(str(wsi_path), resolution=overview_resolution, units=units)

    scale = overview_resolution / patch_resolution
    patch_size_scaled = patch_size * scale
    x_scaled = frame[x_col].to_numpy(dtype=float) * scale
    y_scaled = frame[y_col].to_numpy(dtype=float) * scale

    if background is not None:
        height, width = background.shape[:2]
    else:
        width = int(np.ceil(x_scaled.max() + patch_size_scaled))
        height = int(np.ceil(y_scaled.max() + patch_size_scaled))

    figure_width = 12.0
    figure_height = max(4.0, figure_width * height / max(width, 1))
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    if background is not None:
        ax.imshow(background)
    else:
        ax.set_facecolor("white")

    for x, y, label in zip(x_scaled, y_scaled, frame[label_col].astype(str)):
        color = TISSUE_COLORS.get(label, "#BDBDBD")
        ax.add_patch(
            Rectangle(
                (x, y),
                patch_size_scaled,
                patch_size_scaled,
                facecolor=color,
                edgecolor=color,
                linewidth=0.15,
                alpha=alpha,
            )
        )

    counts = frame[label_col].astype(str).value_counts().to_dict()
    handles = [
        Patch(
            facecolor=TISSUE_COLORS[label],
            edgecolor="#555555",
            label=f"{LABEL_TO_ID[label]} {label}  n={int(counts.get(label, 0))}",
            alpha=0.9,
        )
        for label in TISSUE_ORDER
        if int(counts.get(label, 0)) > 0
    ]
    if handles:
        ax.legend(
            handles=handles,
            title=f"Class\nTotal patches: {len(frame)}",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            frameon=False,
        )

    if title:
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    fig.tight_layout()

    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path
