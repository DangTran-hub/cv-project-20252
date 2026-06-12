# Multi-Object Tracking Project

Vehicle tracking using YOLO11n and ByteTrack on the VNTraffic MOT dataset.

The main workflow is:

```text
input:  one traffic video
level:  baseline, level1, or level2
output: tracking video, MOT prediction txt, and MOT evaluation metrics csv
```

## Project Structure

```text
configs/
  bytetrack_baseline.yaml    Baseline ByteTrack parameters
  bytetrack_custom.yaml      Tuned level 1 ByteTrack parameters
  custom_tracker.yaml        Level 2 custom tracker parameters
  vehicle.yaml               YOLO dataset config

scripts/
  export_baseline_result.py  Export YOLO + ByteTrack result to MOT txt
  evaluate_mot.py            Evaluate prediction txt against MOT ground truth
  render_tracking_video.py   Render bbox and track IDs to video
  tuned_level_1.py           Level 1 tuned tracking pipeline
  tuned_level_2.py           Level 2 local tracker pipeline

trackers/
  basetrack.py               Track state and ID base class
  kalman_filter.py           Local XYAH Kalman filter
  matching.py                IoU, score fusion, and linear assignment helpers
  track.py                   Local ByteTrack lifecycle implementation
  custom_byte_tracker.py     Level 2 ByteTrack variant with custom association
  __init__.py                Local tracker exports

main.py                      Main entrypoint for baseline/level1/level2 pipeline
outputs/
  baseline/                  Baseline outputs
  level1/                    Level 1 outputs from main.py
  level2/                    Level 2 outputs from main.py
  tuned/                     Standalone tuned_level_1.py outputs
```

## Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

If your environment already has a virtual environment in `.venv`, use:

```bash
.venv/bin/python main.py --help
```

## Required Input Files

The default pipeline expects:

```text
yolo11n.pt
dataset/raw/Vehicle_Tracking/VNTraffic/VNTraffic_Original-video.mp4
dataset/raw/Vehicle_Tracking/VNTraffic/VNTraffic_GroundTruth.txt
```

`VNTraffic_GroundTruth.txt` is required only if you want MOT metrics. The rendered tracking video can still be generated from the input video and model.

## Quick Start

Run tuned level 1 on the default VNTraffic video:

```bash
python main.py --level level1 --overwrite
```

Run the baseline:

```bash
python main.py --level baseline --overwrite
```

Run level 2 with the project-local tracker:

```bash
python main.py --level level2 --overwrite
```

The default model is `yolo11n.pt`. Because this is COCO-pretrained, the pipeline keeps only COCO vehicle classes by default:

```text
car, motorcycle, bus, truck
```

## Outputs

For `level1`, `main.py` writes:

```text
outputs/level1/vntraffic_level1_yolo11n_bytetrack.txt
outputs/level1/tables/vntraffic_level1_metrics.csv
outputs/level1/vntraffic_level1_yolo11n_bytetrack.mp4
```

For `level2`, `main.py` writes:

```text
outputs/level2/vntraffic_level2_yolo11n_custom_bytetrack.txt
outputs/level2/tables/vntraffic_level2_metrics.csv
outputs/level2/vntraffic_level2_yolo11n_custom_bytetrack.mp4
```

For `baseline`, `main.py` writes:

```text
outputs/baseline/vntraffic_baseline_yolo11n_bytetrack.txt
outputs/baseline/tables/vntraffic_baseline_metrics.csv
outputs/baseline/vntraffic_baseline_yolo11n_bytetrack.mp4
```

The `.txt` file is MOT-format prediction output. The `.csv` file contains metrics such as MOTA, IDF1, precision, recall, false positives, false negatives, ID switches, and fragmentations. The `.mp4` file is the visualization video with bounding boxes and track IDs.

## Run With A Custom Video

```bash
python main.py \
  --level level1 \
  --video path/to/input_video.mp4 \
  --gt path/to/ground_truth.txt \
  --model yolo11n.pt \
  --overwrite
```

If you do not have ground truth, skip evaluation by running the lower-level rendering script directly:

```bash
python scripts/render_tracking_video.py \
  --model yolo11n.pt \
  --video path/to/input_video.mp4 \
  --tracker configs/bytetrack_custom.yaml \
  --out outputs/visualizations/custom_tracking.mp4 \
  --vehicle_coco \
  --overwrite
```

## Run With A Custom YOLO Vehicle Model

If your model has one class named `vehicle` with class ID `0`, run:

```bash
python main.py \
  --level level1 \
  --model runs/detect/train/weights/best.pt \
  --classes 0 \
  --overwrite
```

To disable class filtering completely:

```bash
python main.py --level level1 --all_classes --overwrite
```

## Levels

### Baseline

Baseline uses:

```text
configs/bytetrack_baseline.yaml
YOLO confidence default: 0.25
```

Run:

```bash
python main.py --level baseline --overwrite
```

### Level 1

Level 1 uses tuned ByteTrack parameters:

```text
configs/bytetrack_custom.yaml
YOLO confidence default: 0.1
COCO vehicle class filter enabled by default
```

Run:

```bash
python main.py --level level1 --overwrite
```

You can also run the level 1 script directly:

```bash
python scripts/tuned_level_1.py --overwrite
```

For quick metric tuning without rendering video:

```bash
python scripts/tuned_level_1.py --skip_render
```

### Level 2

Level 2 uses a local ByteTrack variant instead of `model.track(...)`.
YOLO runs as a detector, then detections are passed to `trackers/Level2BYTETracker`.

The custom association cost combines:

```text
IoU cost
center-distance cost
bbox shape-change cost
detection confidence cost
class mismatch penalty
```

Level 2 defaults:

```text
configs/custom_tracker.yaml
YOLO confidence default: 0.08
COCO vehicle class filter enabled by default
```

Run:

```bash
python main.py --level level2 --overwrite
```

Run the level 2 script directly:

```bash
python scripts/tuned_level_2.py --overwrite
```

For quick metric tuning without rendering video:

```bash
python scripts/tuned_level_2.py --skip_render
```

## Useful Options

```bash
python main.py --level level2 --show_conf --show_class --overwrite
```

```bash
python main.py --level level2 --max_frames 100 --overwrite
```

```bash
python main.py --level level2 --conf 0.08 --iou 0.5 --overwrite
```

## Current Level 1 Result On VNTraffic

Using `yolo11n.pt` and the default VNTraffic input:

```text
Baseline MOTA: 42.88%
Level 1 MOTA: 43.54%

Baseline IDF1: 56.21%
Level 1 IDF1: 57.99%

Baseline ID switches: 55
Level 1 ID switches: 40
```

The corresponding level 1 result files are in:

```text
outputs/level1/
```

## Current Level 2 Result On VNTraffic

Using `yolo11n.pt` and the default VNTraffic input:

```text
Baseline MOTA: 42.88%
Level 1 MOTA: 43.54%
Level 2 MOTA: 45.15%

Baseline IDF1: 56.21%
Level 1 IDF1: 57.99%
Level 2 IDF1: 59.98%

Baseline ID switches: 55
Level 1 ID switches: 40
Level 2 ID switches: 42
```

Level 2 improves MOTA and IDF1 over both earlier levels. It has fewer ID
switches than baseline, but level 1 remains slightly better on ID switches.
The level 2 result files are in `outputs/level2/`.
