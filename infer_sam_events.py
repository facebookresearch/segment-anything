import os
import json
import argparse
import cv2
import numpy as np
import torch

from segment_anything import sam_model_registry, SamPredictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAM inference using bbox prompts and event annotations"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-type", type=str, default="vit_h",
                        choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--annotation", type=str, default="annotations.json")
    parser.add_argument("--images-dir", type=str, default="images")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda")

    # 🔹 NEW
    parser.add_argument(
        "--event-id",
        type=str,
        default=None,
        help="Run inference for only this event (e.g. event1)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    annotation_path = os.path.join(base_dir, args.annotation)
    images_dir = os.path.join(base_dir, args.images_dir)
    output_dir = os.path.join(base_dir, args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Load SAM
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(args.device)
    predictor = SamPredictor(sam)

    # Load annotations
    with open(annotation_path, "r") as f:
        annotations = json.load(f)

    # 🔹 Filter events
    if args.event_id:
        if args.event_id not in annotations:
            raise ValueError(f"Event '{args.event_id}' not found in annotation")
        annotations = {args.event_id: annotations[args.event_id]}

    # Inference
    for event_id, event_data in annotations.items():
        print(f"\nProcessing {event_id}: {event_data['event_name']}")

        event_out_dir = os.path.join(output_dir, event_id)
        os.makedirs(event_out_dir, exist_ok=True)

        frames = event_data["frames"]
        if isinstance(frames, dict):
            frames = [frames]

        for frame_data in frames:
            frame_no = frame_data["frame_number"]
            image_path = os.path.join(images_dir, f"frame_{frame_no}.jpg")

            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(image_path)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)

            for obj_name, obj_data in frame_data["objects"].items():
                box = np.array(obj_data["bbox"], dtype=np.float32)

                masks, scores, _ = predictor.predict(
                    box=box,
                    multimask_output=False
                )

                mask = masks[0].astype(np.uint8) * 255
                out_path = os.path.join(
                    event_out_dir, f"frame_{frame_no}_{obj_name}.png"
                )
                cv2.imwrite(out_path, mask)

                print(f"  Saved → {out_path}")

    print("\nDone ✅")


if __name__ == "__main__":
    main()
