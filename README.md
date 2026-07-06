# RF-DETR Object Detection — Aquarium Dataset

Final project: an end-to-end object detection system built by fine-tuning **RF-DETR** (Receptive-Field-Enhanced Detection Transformer) on the [Aquarium Dataset](https://universe.roboflow.com/brad-dwyer/aquarium-combined) (Roboflow Universe).

## Overview

This repository contains a complete pipeline for:
- Acquiring and preparing an object detection dataset (COCO format)
- Fine-tuning RF-DETR (DINOv2 backbone + deformable transformer encoder-decoder + Hungarian matching)
- Training with AdamW, cosine LR scheduling, early stopping, and checkpointing
- Evaluating with precision, recall, classification accuracy, IoU-based matching, and mAP@0.5 / mAP@0.5:0.95
- Visualizing predictions against ground truth

## Dataset

| | |
|---|---|
| Source | [Aquarium Dataset, Roboflow Universe](https://universe.roboflow.com/brad-dwyer/aquarium-combined) |
| Images | 638 |
| Classes (7) | fish, jellyfish, penguin, puffin, shark, starfish, stingray |
| Format | COCO JSON, pre-split into `train/`, `valid/`, `test/` |

## Repository Structure

```
.
├── RF_DETR_Aquarium_Object_Detection.ipynb   # Main notebook: data prep, training, evaluation, visualization
├── README.md                                 # This file
├── requirements.txt                          # Pinned dependencies
├── outputs/
│   ├── checkpoint_best_ema.pth               # Best fine-tuned weights (EMA) — add after training
│   ├── coco_predictions.json                 # Raw predictions in COCO results format
│   └── prediction_visualizations.png         # Sample ground-truth vs. prediction overlays
└── report/
    └── RF_DETR_Project_Report.docx           # Final written report
```

## Setup

Requires Python ≥ 3.10 and a CUDA-capable GPU (an NVIDIA T4 or better; free-tier Google Colab GPU is sufficient).

```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_NAME>
pip install -r requirements.txt
```

Or, to reproduce exactly as run here, open `RF_DETR_Aquarium_Object_Detection.ipynb` in Google Colab, set the runtime to a GPU (Runtime → Change runtime type → T4 GPU), and run cells top to bottom.

You will need a free [Roboflow API key](https://app.roboflow.com/settings/api) to download the dataset — the notebook prompts for it.

## Running the Pipeline

1. **Data acquisition** — the notebook downloads the Aquarium dataset directly via the Roboflow SDK in COCO format, already split into train/valid/test.
2. **Model** — loads `RFDETRBase` (pretrained on COCO) from the open-source [`rfdetr`](https://github.com/roboflow/rf-detr) package.
3. **Training** — run the training cell:
   ```python
   model.train(
       dataset_dir=str(DATASET_DIR),
       epochs=30,
       batch_size=4,
       grad_accum_steps=4,
       lr=1e-4,
       lr_scheduler="cosine",
       early_stopping=True,
       early_stopping_patience=5,
       output_dir="/content/rfdetr_output",
   )
   ```
4. **Evaluation** — run the evaluation cells to compute precision, recall, classification accuracy, and mAP (via `pycocotools`).
5. **Visualization** — run the visualization cell to generate `prediction_visualizations.png`.

## Results

| Metric | Value |
|---|---|
| Precision (test set) | 0.898 |
| Recall (test set) | 0.765 |
| Classification accuracy (matched boxes, test set) | 0.987 |
| mAP@0.5 (validation set) | 0.823 |
| mAP@0.5:0.95 (validation set) | 0.542 |

Note: test-set mAP via `pycocotools` was not obtained — the model's predicted class indices did not align cleanly with the dataset's category list, and this was not resolved within the project timeline. mAP above is reported from the validation-set metrics logged automatically by `rfdetr` during training instead. See `report/RF_DETR_Project_Report.docx` for full discussion, per-class breakdown, and error analysis.

## Implementation Notes

- We fine-tune the open-source `rfdetr` package rather than reimplementing DETR/RF-DETR from raw PyTorch, per the assignment's recommended "adaptation" strategy. The package implements all required architectural components: CNN/ViT backbone, transformer encoder-decoder, receptive-field enhancement (multi-scale deformable attention), FFN prediction heads, and a Hungarian matcher for training-time bipartite matching.
- IoU computation and TP/FP/FN matching for precision/recall/accuracy are implemented from scratch in the notebook (see `compute_iou` and `match_detections`).
- mAP@0.5 and mAP@0.5:0.95 are computed using `pycocotools.cocoeval.COCOeval`, following the standard COCO evaluation protocol.

## References

- Carion, N., et al. (2020). *End-to-End Object Detection with Transformers (DETR)*. ECCV.
- Zhu, X., et al. (2021). *Deformable DETR: Deformable Transformers for End-to-End Object Detection*. ICLR.
- Robinson, I., Robicheaux, P., Popov, M., Ramanan, D., & Peri, N. (2025). *RF-DETR: Neural Architecture Search for Real-Time Detection Transformers*. Roboflow.
- Oquab, M., et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision*.
- Lin, T.-Y., et al. (2014). *Microsoft COCO: Common Objects in Context*.

## License

Add your group's chosen license here (MIT is common for academic projects).
