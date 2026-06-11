"""
Run a tuned ByteTrack experiment end-to-end.

This script reuses the existing project utilities:
- export_baseline_result.py for YOLO + ByteTrack MOT prediction export
- evaluate_mot.py for MOT metrics
- render_tracking_video.py for visualization video rendering

It keeps the original baseline untouched and writes tuned outputs under
outputs/tuned by default.

Usage:
    Run with the project defaults:

        python scripts/run_tuned_bytetrack.py --overwrite

    The default input video is:

        dataset/raw/Vehicle_Tracking/VNTraffic/VNTraffic_Original-video.mp4

    The default YOLO model is:

        yolo11n.pt

    Because the default model is COCO-pretrained YOLO11n and the VNTraffic
    ground truth contains vehicles, the script keeps COCO vehicle classes by
    default: car, motorcycle, bus, and truck. For a custom one-class vehicle
    model, pass --classes 0. To disable class filtering, pass --all_classes.

    The default tuned tracker config is:

        configs/bytetrack_custom.yaml

    The default YOLO confidence threshold is 0.1. This is intentionally lower
    than the baseline 0.25 so ByteTrack can receive weak detections and use
    them for second-stage association instead of losing vehicles too early.

    The script creates three tuned outputs:

        outputs/tuned/vntraffic_tuned_yolo11n_bytetrack.txt
        outputs/tuned/tables/vntraffic_tuned_metrics.csv
        outputs/tuned/vntraffic_tuned_yolo11n_bytetrack.mp4

    Run with a custom model, video, and output paths:

        python scripts/run_tuned_bytetrack.py \
            --model runs/detect/train/weights/best.pt \
            --video dataset/raw/Vehicle_Tracking/VNTraffic/VNTraffic_Original-video.mp4 \
            --gt dataset/raw/Vehicle_Tracking/VNTraffic/VNTraffic_GroundTruth.txt \
            --tracker configs/bytetrack_custom.yaml \
            --classes 0 \
            --pred outputs/tuned/custom_prediction.txt \
            --metrics outputs/tuned/tables/custom_metrics.csv \
            --vis outputs/tuned/custom_visualization.mp4 \
            --overwrite

    Useful quick-tuning command without rendering video:

        python scripts/run_tuned_bytetrack.py --skip_render
"""

from pathlib import Path
import argparse

from evaluate_mot import evaluate_mot
from export_baseline_result import export_baseline_mot
from render_tracking_video import render_tracking_video


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = PROJECT_ROOT / "yolo11n.pt"
DEFAULT_VIDEO = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "Vehicle_Tracking"
    / "VNTraffic"
    / "VNTraffic_Original-video.mp4"
)
DEFAULT_GT = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "Vehicle_Tracking"
    / "VNTraffic"
    / "VNTraffic_GroundTruth.txt"
)
DEFAULT_TRACKER = PROJECT_ROOT / "configs" / "bytetrack_custom.yaml"
DEFAULT_PRED = PROJECT_ROOT / "outputs" / "tuned" / "vntraffic_tuned_yolo11n_bytetrack.txt"
DEFAULT_METRICS = PROJECT_ROOT / "outputs" / "tuned" / "tables" / "vntraffic_tuned_metrics.csv"
DEFAULT_VIDEO_OUT = PROJECT_ROOT / "outputs" / "tuned" / "vntraffic_tuned_yolo11n_bytetrack.mp4"
COCO_VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck


def parse_class_ids(value):
    if value is None:
        return None

    class_ids = []
    for item in value:
        for part in item.split(","):
            part = part.strip()
            if part:
                class_ids.append(int(part))

    return class_ids or None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run tuned YOLO + ByteTrack export, evaluation, and visualization."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLO model path")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="Input video path")
    parser.add_argument("--gt", default=str(DEFAULT_GT), help="MOT ground-truth txt path")
    parser.add_argument("--tracker", default=str(DEFAULT_TRACKER), help="Tuned ByteTrack YAML path")
    parser.add_argument("--pred", default=str(DEFAULT_PRED), help="Output MOT prediction txt")
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS), help="Output metrics CSV")
    parser.add_argument("--vis", default=str(DEFAULT_VIDEO_OUT), help="Output visualization video")
    parser.add_argument("--conf", type=float, default=0.1, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO NMS IoU threshold")
    parser.add_argument("--name", default="tuned", help="Experiment name written to the metrics table")
    parser.add_argument(
        "--skip_export",
        action="store_true",
        help="Reuse an existing MOT prediction txt instead of exporting a new one",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip MOT metric evaluation",
    )
    parser.add_argument(
        "--skip_render",
        action="store_true",
        help="Skip visualization rendering for quick tuning runs",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional class IDs, accepts space or comma separated values, e.g. 2 3 5 7",
    )
    parser.add_argument(
        "--vehicle_coco",
        action="store_true",
        help="Keep only COCO vehicle classes: car, motorcycle, bus, truck. This is the default preset.",
    )
    parser.add_argument(
        "--all_classes",
        action="store_true",
        help="Disable the default COCO vehicle class filter",
    )
    parser.add_argument("--show_class", action="store_true", help="Show class name in video labels")
    parser.add_argument("--show_conf", action="store_true", help="Show confidence score in video labels")
    parser.add_argument("--max_frames", type=int, default=None, help="Render only first N frames")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing visualization video",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.all_classes:
        classes = None
    elif args.classes is not None:
        classes = parse_class_ids(args.classes)
    else:
        classes = COCO_VEHICLE_CLASSES

    if not args.skip_export:
        print("===== Tuned ByteTrack: export MOT prediction =====")
        export_baseline_mot(
            model_path=args.model,
            video_path=args.video,
            tracker_cfg=args.tracker,
            output_txt=args.pred,
            conf=args.conf,
            iou=args.iou,
            classes=classes,
        )
    else:
        print(f"===== Tuned ByteTrack: reuse MOT prediction at {args.pred} =====")

    if not args.skip_eval:
        print("\n===== Tuned ByteTrack: evaluate MOT metrics =====")
        evaluate_mot(
            gt_path=args.gt,
            pred_path=args.pred,
            output_csv=args.metrics,
            name=args.name,
        )
    else:
        print("\n===== Tuned ByteTrack: skip MOT evaluation =====")

    if not args.skip_render:
        print("\n===== Tuned ByteTrack: render visualization video =====")
        render_tracking_video(
            model_path=args.model,
            video_path=args.video,
            tracker_cfg=args.tracker,
            output_video=args.vis,
            save_mot=None,
            conf=args.conf,
            iou=args.iou,
            classes=classes,
            show_class=args.show_class,
            show_conf=args.show_conf,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
        )
    else:
        print("\n===== Tuned ByteTrack: skip visualization rendering =====")


if __name__ == "__main__":
    main()
