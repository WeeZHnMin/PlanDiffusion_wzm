import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image, ImageDraw


TYPE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "living room": (252, 220, 151),
    "master room": (208, 190, 226),
    "common room": (208, 190, 226),
    "kitchen": (173, 222, 192),
    "bathroom": (182, 216, 244),
    "dining room": (248, 203, 173),
}


def parse_args():
    p = argparse.ArgumentParser(description="Visualize converted Tell2Design floorplan json.")
    p.add_argument(
        "--input",
        default="Tell2Design-comp/T5/data/floorplan/floorplan_train.json",
        help="Converted floorplan json path.",
    )
    p.add_argument(
        "--output-dir",
        default="Tell2Design-comp/T5/data/floorplan/vis_train",
        help="Directory to save visualization png files.",
    )
    p.add_argument("--limit", type=int, default=16, help="How many samples to render.")
    p.add_argument("--canvas", type=int, default=256, help="Canvas size.")
    return p.parse_args()


def room_color(room_type: str) -> Tuple[int, int, int]:
    for prefix, color in TYPE_COLORS.items():
        if room_type.startswith(prefix):
            return color
    return (220, 220, 220)


def draw_sample(sample: dict, canvas: int) -> Image.Image:
    image = Image.new("RGB", (canvas, canvas), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    for room in sample.get("rooms", []):
        x0 = int(room["x_min"])
        y0 = int(room["y_min"])
        x1 = int(room["x_max"])
        y1 = int(room["y_max"])
        color = room_color(room["room_type"])
        draw.rectangle([x0, y0, x1, y1], fill=color, outline=(90, 90, 90), width=2)
        draw.text((x0 + 4, y0 + 4), str(room["room_type"]), fill=(30, 30, 30))

    for point in sample.get("boundary", []):
        if len(point) >= 2:
            x = int(point[0])
            y = int(point[1])
            if 0 <= x < canvas and 0 <= y < canvas:
                image.putpixel((x, y), (0, 0, 0))

    for box in sample.get("boundary_boxs", []):
        x0 = int(box["x_min"])
        y0 = int(box["y_min"])
        x1 = int(box["x_max"])
        y1 = int(box["y_max"])
        outline = (255, 80, 80) if box.get("room_type") == "negative" else (40, 40, 40)
        draw.rectangle([x0, y0, x1, y1], outline=outline, width=2)

    title = str(sample.get("img_id", "unknown"))
    draw.text((6, 6), title, fill=(0, 0, 0))
    return image


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = json.loads(input_path.read_text(encoding="utf-8"))
    for idx, sample in enumerate(samples[: args.limit]):
        image = draw_sample(sample, args.canvas)
        img_id = str(sample.get("img_id", f"sample_{idx:05d}"))
        image.save(output_dir / f"{idx:04d}_{img_id}.png")

    print(f"saved {min(len(samples), args.limit)} images to {output_dir}")


if __name__ == "__main__":
    main()
