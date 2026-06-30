# Minimal WSI-to-SpatialNI Example

This example follows the repository order:

1. WSI preprocessing
2. patch classification
3. SpatialNI feature construction
4. WSI patch class map export

By default, `classify_wsi_with_tiatoolbox()` performs Reinhard stain
normalization. You must provide a reference H&E patch image through
`reference_image`.

```python
from pathlib import Path

from patch_classification import classify_wsi_with_tiatoolbox
from spatialni_feature_engine import (
    build_patient_spatialni_from_predictions,
    plot_wsi_patch_class_map,
)

project_dir = Path(__file__).resolve().parent

wsi_path = project_dir / "wsi" / "TCGA-A6-2676-01Z-00-DX1.c465f6e0-b47c-48e9-bdb1-67077bb16c67.svs"
reference_image = project_dir / "reference" / "reference_patch.jpg"
output_dir = project_dir / "outputs" / "TCGA-A6-2676-01"
output_dir.mkdir(parents=True, exist_ok=True)

predictions = classify_wsi_with_tiatoolbox(
    wsi_path=wsi_path,
    output_csv=output_dir / "patch_predictions.csv",
    batch_size=16,
    device="cuda",
    reference_image=reference_image,
)

spatialni = build_patient_spatialni_from_predictions(
    predictions,
    patient_id="TCGA-A6-2676-01",
)
spatialni.to_csv(output_dir / "patient_spatialni.csv", index=False)

plot_wsi_patch_class_map(
    predictions,
    output_png=output_dir / "wsi_patch_class_map.png",
    wsi_path=wsi_path,
    title="TCGA-A6-2676-01 patch class map",
)
```

If no stain-normalization reference image is available and you only want to test
that the example slide can run end-to-end, explicitly disable normalization:

```python
predictions = classify_wsi_with_tiatoolbox(
    wsi_path=wsi_path,
    output_csv=output_dir / "patch_predictions.csv",
    batch_size=16,
    device="cuda",
    stain_normalization=False,
)
```
