"""
Frame Extraction Utility for VNTraffic MOT Dataset (Zenodo).

This script extracts annotated frames from the original VNTraffic video based on
the frame indices provided in the MOT-format ground truth file. Each frame ID is
read from `VNTraffic_GroundTruth.txt`, and the corresponding frame is extracted
from `VNTraffic_Original-video.mp4` using OpenCV.

The extracted images are saved with a dataset-specific filename prefix, such as
`zenodo_vntraffic_000000.jpg`, to avoid name collisions when combining this
dataset with other custom datasets. This naming convention ensures that each
image can be uniquely identified while preserving the original frame index for
accurate synchronization with the ground truth annotations.

Input:
    - Original video file: VNTraffic_Original-video.mp4
    - MOT ground truth file: VNTraffic_GroundTruth.txt

Output:
    - Extracted image frames in JPG format, stored in the YOLO dataset image
      directory, e.g. dataset/processed/train/images/

Note:
    The script assumes that the frame IDs in the ground truth file are zero-based.
    Therefore, frame_id = 0 corresponds to the first frame of the video.
DangTran
"""


from pathlib import Path
import cv2


# =========================
# Project paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "dataset" / "raw" / "Vehicle_Tracking" / "VNTraffic"

VIDEO_PATH = RAW_DIR / "VNTraffic_Original-video.mp4"
GT_PATH = RAW_DIR / "VNTraffic_GroundTruth.txt"

OUTPUT_IMAGES_DIR = PROJECT_ROOT / "dataset" / "processed" / "train" / "images"

DATASET_PREFIX = "zenodo_vntraffic"


def load_frame_ids(gt_file: Path):
    """
    Read unique frame IDs from MOT ground truth file.

    MOT format:
    frame_id, object_id, x, y, w, h, confidence, -1, -1, -1
    """
    frame_ids = set()

    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            cols = line.split(",")

            if len(cols) < 6:
                continue

            frame_id = int(float(cols[0]))
            frame_ids.add(frame_id)

    return sorted(frame_ids)


def extract_frames(video_path: Path, frame_ids, output_dir: Path, prefix: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video path: {video_path}")
    print(f"Total video frames reported by OpenCV: {total_video_frames}")
    print(f"Need to extract {len(frame_ids)} annotated frames")

    if frame_ids:
        max_gt_frame = max(frame_ids)
        print(f"Max frame_id in GT: {max_gt_frame}")

        if max_gt_frame >= total_video_frames:
            print(
                "[WARNING] Max GT frame_id is greater than or equal to total video frames. "
                "Please check whether GT uses 0-based or 1-based indexing."
            )

    frame_id_set = set(frame_ids)

    current_frame_id = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if current_frame_id in frame_id_set:
            image_name = f"{prefix}_{current_frame_id:06d}.jpg"
            image_path = output_dir / image_name

            success = cv2.imwrite(str(image_path), frame)

            if not success:
                print(f"[WARNING] Failed to save: {image_path}")
            else:
                saved_count += 1

        current_frame_id += 1

    cap.release()

    print("\nDone.")
    print(f"Saved {saved_count} frames to: {output_dir}")

    missing = len(frame_ids) - saved_count
    if missing > 0:
        print(f"[WARNING] Missing {missing} frames. Check video length and GT frame_id.")


if __name__ == "__main__":
    frame_ids = load_frame_ids(GT_PATH)

    extract_frames(
        video_path=VIDEO_PATH,
        frame_ids=frame_ids,
        output_dir=OUTPUT_IMAGES_DIR,
        prefix=DATASET_PREFIX,
    )