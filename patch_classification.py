from __future__ import annotations

"""
Step 02 - Patch classification.

Run this after WSI preprocessing. The main public entry point is
`classify_wsi_with_tiatoolbox`, which follows the TIAToolbox PatchPredictor
workflow and applies Reinhard stain normalization by default.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.color import lab2rgb, rgb2lab
from torch import nn
from torchvision import models


KATHER_LABELS = [
    "BACK",
    "NORM",
    "DEB",
    "TUM",
    "ADI",
    "MUC",
    "MUS",
    "STR",
    "LYM",
]
DEFAULT_REFERENCE_IMAGE: Path | None = None
DEFAULT_WEIGHTS = Path.home() / ".tiatoolbox" / "models" / "wide_resnet101_2-kather100k.pth"

PATCH_NAME_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<cls>\d+)_(?P<x>\d+)_(?P<y>\d+)_?(?P<dup> \(\d+\))?\.jpg$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PredictionRecord:
    input_name: str
    output_name: str
    predicted_class_id: int
    predicted_class_label: str
    probabilities: np.ndarray | None


def load_rgb(path: str | Path) -> np.ndarray:
    file_path = Path(path)
    buffer = np.fromfile(file_path, dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {file_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_rgb_jpeg(array: np.ndarray, path: str | Path, quality: int = 95) -> None:
    file_path = Path(path)
    bgr = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError(f"Failed to encode image: {file_path}")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(file_path)


def _require_tiatoolbox_predictor():
    try:
        from tiatoolbox.models.engine.patch_predictor import IOPatchPredictorConfig, PatchPredictor
    except ImportError as exc:
        raise ImportError("TIAToolbox is required for WSI-level patch classification.") from exc
    return IOPatchPredictorConfig, PatchPredictor


def _require_wsi_helpers():
    try:
        from wsi_preprocessing import notebook_style_patch_locations, open_wsi
    except ImportError:
        from .wsi_preprocessing import notebook_style_patch_locations, open_wsi
    return notebook_style_patch_locations, open_wsi


def _format_probabilities(probabilities: Sequence[float] | np.ndarray | None) -> str:
    if probabilities is None:
        return ""
    return ";".join(f"{float(value):.8f}" for value in probabilities)


class ReinhardNormalizer:
    def __init__(self, lab_backend: str = "cv2") -> None:
        if lab_backend not in {"cv2", "skimage"}:
            raise ValueError("lab_backend must be 'cv2' or 'skimage'.")
        self.lab_backend = lab_backend
        self.target_means: np.ndarray | None = None
        self.target_stds: np.ndarray | None = None

    def _rgb_to_lab(self, image: np.ndarray) -> np.ndarray:
        if self.lab_backend == "cv2":
            return cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        return rgb2lab(image.astype(np.float32) / 255.0).astype(np.float32)

    def _lab_to_rgb(self, image: np.ndarray) -> np.ndarray:
        if self.lab_backend == "cv2":
            clipped = np.clip(image, 0, 255).astype(np.uint8)
            return cv2.cvtColor(clipped, cv2.COLOR_LAB2RGB)
        restored = image.copy()
        restored[..., 0] = np.clip(restored[..., 0], 0.0, 100.0)
        rgb = lab2rgb(restored)
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    def fit(self, target: np.ndarray) -> None:
        target_lab = self._rgb_to_lab(target)
        flat = target_lab.reshape(-1, 3)
        self.target_means = flat.mean(axis=0)
        self.target_stds = np.maximum(flat.std(axis=0), 1e-6)

    def transform(self, image: np.ndarray) -> np.ndarray:
        if self.target_means is None or self.target_stds is None:
            raise RuntimeError("Normalizer must be fit before transform.")
        source_lab = self._rgb_to_lab(image)
        flat = source_lab.reshape(-1, 3)
        source_means = flat.mean(axis=0)
        source_stds = np.maximum(flat.std(axis=0), 1e-6)
        normalized = (flat - source_means) / source_stds
        shifted = normalized * self.target_stds + self.target_means
        return self._lab_to_rgb(shifted.reshape(source_lab.shape))


class KatherWideResNet(nn.Module):
    def __init__(self, num_classes: int = 9) -> None:
        super().__init__()
        backbone = models.wide_resnet101_2(weights=None)
        self.feat_extract = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        channels = self.feat_extract(torch.rand([2, 3, 96, 96])).shape[1]
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feat_extract(images)
        pooled = self.pool(features)
        pooled = torch.flatten(pooled, 1)
        logits = self.classifier(pooled)
        return torch.softmax(logits, dim=-1)


def build_model(weights_path: str | Path, device: str = "cpu") -> nn.Module:
    weights_file = Path(weights_path)
    if not weights_file.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_file}")
    model = KatherWideResNet(num_classes=len(KATHER_LABELS))
    state_dict = torch.load(weights_file, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def iter_patch_paths(input_dir: str | Path, limit: int | None = None) -> list[Path]:
    patch_paths = sorted(path for path in Path(input_dir).glob("*.jpg") if path.is_file())
    return patch_paths if limit is None else patch_paths[:limit]


def build_model_input_batch(images: list[np.ndarray]) -> torch.Tensor:
    batch_arrays: list[np.ndarray] = []
    for image in images:
        resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        chw = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))
        batch_arrays.append(chw)
    return torch.from_numpy(np.stack(batch_arrays, axis=0))


def predict_probabilities(
    model: nn.Module,
    normalized_images: list[np.ndarray],
    device: str = "cpu",
) -> np.ndarray:
    with torch.inference_mode():
        batch = build_model_input_batch(normalized_images).to(device=device, dtype=torch.float32, non_blocking=True)
        return model(batch).detach().cpu().numpy()


def build_patch_identity(path_or_name: str | Path) -> tuple[str, str, str, str]:
    patch_name = Path(path_or_name).name if not isinstance(path_or_name, str) else Path(path_or_name).name
    match = PATCH_NAME_RE.match(patch_name)
    if match is None:
        raise ValueError(f"Unsupported patch file name format: {patch_name}")
    return (
        match.group("prefix"),
        match.group("x"),
        match.group("y"),
        match.group("dup") or "",
    )


def build_renamed_path(path: str | Path, predicted_class: int) -> Path:
    file_path = Path(path)
    match = PATCH_NAME_RE.match(file_path.name)
    if match is None:
        raise ValueError(f"Unsupported patch file name format: {file_path.name}")
    new_name = (
        f"{match.group('prefix')}_{predicted_class}_{match.group('x')}_"
        f"{match.group('y')}_{match.group('dup') or ''}.jpg"
    )
    return file_path.with_name(new_name)


def load_existing_records(
    input_paths: list[Path],
    output_dir: str | Path,
) -> tuple[list[PredictionRecord], set[tuple[str, str, str, str]]]:
    output_path = Path(output_dir)
    if not output_path.exists():
        return [], set()

    input_by_identity = {build_patch_identity(path): path for path in input_paths}
    records: list[PredictionRecord] = []
    processed_keys: set[tuple[str, str, str, str]] = set()

    for saved_path in sorted(output_path.glob("*.jpg")):
        try:
            identity = build_patch_identity(saved_path)
        except ValueError:
            continue
        input_path = input_by_identity.get(identity)
        if input_path is None:
            continue
        match = PATCH_NAME_RE.match(saved_path.name)
        if match is None:
            continue
        predicted_class = int(match.group("cls"))
        processed_keys.add(identity)
        records.append(
            PredictionRecord(
                input_name=input_path.name,
                output_name=saved_path.name,
                predicted_class_id=predicted_class,
                predicted_class_label=KATHER_LABELS[predicted_class],
                probabilities=None,
            )
        )

    return records, processed_keys


def write_manifest(path: str | Path, records: Iterable[PredictionRecord]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "input_name",
        "output_name",
        "predicted_class_id",
        "predicted_class_label",
        "probabilities",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "input_name": record.input_name,
                    "output_name": record.output_name,
                    "predicted_class_id": record.predicted_class_id,
                    "predicted_class_label": record.predicted_class_label,
                    "probabilities": "" if record.probabilities is None else ";".join(f"{value:.8f}" for value in record.probabilities.tolist()),
                }
            )


def normalize_images(
    normalizer: ReinhardNormalizer,
    input_paths: list[Path],
) -> list[np.ndarray]:
    return [normalizer.transform(load_rgb(path)) for path in input_paths]


def classify_patch_arrays(
    patch_images: list[np.ndarray],
    patch_names: Sequence[str],
    reference_image: str | Path | np.ndarray | None = DEFAULT_REFERENCE_IMAGE,
    weights_path: str | Path = DEFAULT_WEIGHTS,
    batch_size: int = 32,
    device: str = "cpu",
    lab_backend: str = "cv2",
) -> pd.DataFrame:
    if len(patch_images) != len(patch_names):
        raise ValueError("patch_images and patch_names must have the same length.")
    if not patch_images:
        return pd.DataFrame(columns=["input_name", "predicted_class_id", "predicted_class_label", "probabilities"])
    if reference_image is None:
        raise ValueError("reference_image is required when classifying normalized patch arrays.")

    normalizer = ReinhardNormalizer(lab_backend=lab_backend)
    if isinstance(reference_image, np.ndarray):
        normalizer.fit(reference_image)
    else:
        normalizer.fit(load_rgb(reference_image))
    model = build_model(weights_path=weights_path, device=device)

    records: list[PredictionRecord] = []
    for batch_start in range(0, len(patch_images), batch_size):
        batch_images = patch_images[batch_start : batch_start + batch_size]
        batch_names = patch_names[batch_start : batch_start + batch_size]
        normalized_images = [normalizer.transform(image) for image in batch_images]
        probabilities = predict_probabilities(model=model, normalized_images=normalized_images, device=device)
        predicted_classes = probabilities.argmax(axis=1)

        for input_name, predicted_class, probability in zip(batch_names, predicted_classes, probabilities):
            records.append(
                PredictionRecord(
                    input_name=input_name,
                    output_name="",
                    predicted_class_id=int(predicted_class),
                    predicted_class_label=KATHER_LABELS[int(predicted_class)],
                    probabilities=probability.astype(float),
                )
            )

    return pd.DataFrame(
        [
            {
                "input_name": record.input_name,
                "predicted_class_id": record.predicted_class_id,
                "predicted_class_label": record.predicted_class_label,
                "probabilities": ";".join(f"{value:.8f}" for value in record.probabilities.tolist()) if record.probabilities is not None else "",
            }
            for record in records
        ]
    )


def classify_wsi_with_tiatoolbox(
    wsi_path: str | Path,
    output_csv: str | Path | None = None,
    pretrained_model: str = "wide_resnet101_2-kather100k",
    batch_size: int = 16,
    device: str = "cuda",
    patch_size: tuple[int, int] = (512, 512),
    stride: tuple[int, int] = (512, 512),
    resolution: float = 20.0,
    units: str = "power",
    input_mask: str = "otsu",
    return_probabilities: bool = True,
    stain_normalization: bool = True,
    reference_image: str | Path | None = None,
    lab_backend: str = "cv2",
) -> pd.DataFrame:
    """
    Classify WSI patches with the original notebook-style TIAToolbox workflow.

    Default behavior applies Reinhard stain normalization before classification,
    matching the manuscript Methods. A `reference_image` must therefore be supplied
    unless `stain_normalization=False` is explicitly requested.
    """
    IOPatchPredictorConfig, PatchPredictor = _require_tiatoolbox_predictor()
    notebook_style_patch_locations, open_wsi = _require_wsi_helpers()

    locations = notebook_style_patch_locations(
        wsi_path,
        patch_size=patch_size,
        stride=stride,
        resolution=resolution,
        units=units,
        input_mask=input_mask,
    )

    predictor = PatchPredictor(pretrained_model=pretrained_model, batch_size=batch_size)

    if stain_normalization:
        if reference_image is None:
            raise ValueError("reference_image is required when stain_normalization=True.")
        normalizer = ReinhardNormalizer(lab_backend=lab_backend)
        normalizer.fit(load_rgb(reference_image))
        reader = open_wsi(str(wsi_path))
        predictions: list[int] = []
        probabilities_out: list[str] = []

        for batch_start in range(0, len(locations), batch_size):
            batch_locations = locations.iloc[batch_start : batch_start + batch_size]
            patch_images: list[np.ndarray] = []
            for _, row in batch_locations.iterrows():
                x = int(row["x"])
                y = int(row["y"])
                patch = reader.read_bounds(
                    bounds=[x, y, x + patch_size[0], y + patch_size[1]],
                    resolution=resolution,
                    units=units,
                    coord_space="resolution",
                )
                patch_images.append(normalizer.transform(np.asarray(patch)))

            batch_output = predictor.predict(
                imgs=patch_images,
                mode="patch",
                patch_input_shape=patch_size,
                return_probabilities=return_probabilities,
                device=device,
            )
            batch_predictions = batch_output["predictions"] if isinstance(batch_output, dict) else batch_output
            predictions.extend(int(value) for value in batch_predictions)
            if return_probabilities and isinstance(batch_output, dict):
                probabilities_out.extend(_format_probabilities(row) for row in batch_output.get("probabilities", []))
            else:
                probabilities_out.extend([""] * len(batch_predictions))
    else:
        wsi_ioconfig = IOPatchPredictorConfig(
            input_resolutions=[{"units": units, "resolution": resolution}],
            patch_input_shape=list(patch_size),
            stride_shape=list(stride),
        )
        wsi_output = predictor.predict(
            imgs=[Path(wsi_path)],
            mode="wsi",
            merge_predictions=False,
            ioconfig=wsi_ioconfig,
            return_probabilities=return_probabilities,
            save_dir=None,
            device=device,
        )
        predictions = [int(value) for value in wsi_output[0]["predictions"]]
        if return_probabilities:
            probabilities_out = [_format_probabilities(row) for row in wsi_output[0].get("probabilities", [])]
        else:
            probabilities_out = [""] * len(predictions)

    if len(predictions) != len(locations):
        raise RuntimeError(
            f"Prediction count ({len(predictions)}) does not match patch locations ({len(locations)})."
        )

    result = locations.copy()
    result["predicted_class_id"] = predictions
    result["predicted_class_label"] = [KATHER_LABELS[class_id] for class_id in predictions]
    result["probabilities"] = probabilities_out
    result["stain_normalization"] = bool(stain_normalization)
    result["reference_image"] = "" if reference_image is None else str(reference_image)

    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)

    return result


def classify_patch_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    reference_image: str | Path | None = DEFAULT_REFERENCE_IMAGE,
    weights_path: str | Path = DEFAULT_WEIGHTS,
    batch_size: int = 32,
    device: str = "cpu",
    quality: int = 95,
    limit: int | None = None,
    overwrite: bool = False,
    lab_backend: str = "cv2",
) -> pd.DataFrame:
    input_paths = iter_patch_paths(input_dir, limit=limit)
    if not input_paths:
        return pd.DataFrame(columns=["input_name", "output_name", "predicted_class_id", "predicted_class_label", "probabilities"])
    if reference_image is None:
        raise ValueError("reference_image is required when classifying normalized patch directories.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    normalizer = ReinhardNormalizer(lab_backend=lab_backend)
    normalizer.fit(load_rgb(reference_image))
    model = build_model(weights_path=weights_path, device=device)

    existing_records, processed_keys = load_existing_records(input_paths, output_path)
    pending_paths = [path for path in input_paths if build_patch_identity(path) not in processed_keys]
    records = list(existing_records)

    for batch_start in range(0, len(pending_paths), batch_size):
        batch_paths = pending_paths[batch_start : batch_start + batch_size]
        normalized_images = normalize_images(normalizer, batch_paths)
        probabilities = predict_probabilities(model=model, normalized_images=normalized_images, device=device)
        predicted_classes = probabilities.argmax(axis=1)

        for input_path, normalized_image, predicted_class, probability in zip(
            batch_paths,
            normalized_images,
            predicted_classes,
            probabilities,
        ):
            renamed_path = output_path / build_renamed_path(input_path, int(predicted_class)).name
            if renamed_path.exists() and overwrite:
                renamed_path.unlink()
            save_rgb_jpeg(normalized_image, renamed_path, quality=quality)
            records.append(
                PredictionRecord(
                    input_name=input_path.name,
                    output_name=renamed_path.name,
                    predicted_class_id=int(predicted_class),
                    predicted_class_label=KATHER_LABELS[int(predicted_class)],
                    probabilities=probability.astype(float),
                )
            )

    write_manifest(output_path / "prediction_manifest.csv", records)
    return pd.DataFrame(
        [
            {
                "input_name": record.input_name,
                "output_name": record.output_name,
                "predicted_class_id": record.predicted_class_id,
                "predicted_class_label": record.predicted_class_label,
                "probabilities": None if record.probabilities is None else ";".join(f"{value:.8f}" for value in record.probabilities.tolist()),
            }
            for record in records
        ]
    )


def summarize_patch_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    required = {"predicted_class_label", "predicted_class_id"}
    missing = required.difference(pred_df.columns)
    if missing:
        raise KeyError(f"Missing required prediction columns: {sorted(missing)}")
    return (
        pred_df.groupby(["predicted_class_id", "predicted_class_label"], as_index=False)
        .agg(n_patches=("predicted_class_id", "size"))
        .sort_values("n_patches", ascending=False)
        .reset_index(drop=True)
    )
