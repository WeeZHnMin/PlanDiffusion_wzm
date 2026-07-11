import argparse
import ast
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from tqdm import tqdm


ROOM_TYPE_ORDER = [
    "bathroom",
    "bedroom",
    "living_room",
    "kitchen",
    "corridor",
    "dining_room",
    "other",
]

SUPPORTED_BASE_TYPES = {
    "bathroom",
    "bedroom",
    "living_room",
    "kitchen",
    "corridor",
    "dining_room",
}

DEFAULT_ALL_INPUT = "data/jsonl/graph_160k_spatial.jsonl"
DEFAULT_TRAIN_INPUT = "data/jsonl/graph_160k_spatial_train.jsonl"
DEFAULT_VAL_INPUT = "data/jsonl/graph_160k_spatial_val.jsonl"
DEFAULT_COMBO_VOCAB = "Tell2Design-comp/type_combo_vocab_v3.json"
DEFAULT_OUTPUT_DIR = "Tell2Design-comp/T5/data/floorplan"
WORKER_ID_TO_COMBO: Optional[Dict[int, List[str]]] = None
WORKER_DROP_TYPES: Optional[set] = None


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def load_vocab(path: Path) -> Dict[int, List[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "id_to_combo" in payload:
        return {int(k): list(v) for k, v in payload["id_to_combo"].items()}
    combo_to_id = payload["combo_to_id"]
    base_names = payload["base_type_names"]
    id_to_combo: Dict[int, List[str]] = {}
    for combo_str, cid in combo_to_id.items():
        type_ids = ast.literal_eval(combo_str)
        id_to_combo[int(cid)] = [base_names[str(tid)] for tid in type_ids]
    return id_to_combo


def _build_sorted_neighbors(
    coords: List[Tuple[float, float]],
    adj: List[List[int]],
) -> Dict[int, List[int]]:
    n = len(coords)
    nbrs: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and adj[i][j] == 1:
                nbrs[i].append(j)
    for i in range(n):
        nbrs[i] = sorted(
            nbrs[i],
            key=lambda w: math.atan2(
                coords[w][1] - coords[i][1],
                coords[w][0] - coords[i][0],
            ),
        )
    return nbrs


def _next_half_edge(u: int, v: int, sorted_nbrs: Dict[int, List[int]]) -> Optional[int]:
    nbrs = sorted_nbrs[v]
    if not nbrs:
        return None
    idx = nbrs.index(u)
    return nbrs[(idx - 1) % len(nbrs)]


def _signed_area(face: List[int], coords: List[Tuple[float, float]]) -> float:
    pts = [coords[i] for i in face]
    area = 0.0
    for i, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def find_faces(coords: List[Tuple[float, float]], adj: List[List[int]]) -> List[List[int]]:
    n = len(coords)
    sorted_nbrs = _build_sorted_neighbors(coords, adj)
    visited = set()
    faces: List[List[int]] = []

    for u in range(n):
        for v in sorted_nbrs[u]:
            if (u, v) in visited:
                continue
            face: List[int] = []
            cu, cv = u, v
            steps = 0
            while (cu, cv) not in visited and steps < n * n:
                visited.add((cu, cv))
                face.append(cu)
                nw = _next_half_edge(cu, cv, sorted_nbrs)
                if nw is None:
                    break
                cu, cv = cv, nw
                steps += 1
            if len(face) >= 3:
                faces.append(face)

    if not faces:
        return []

    abs_areas = [abs(_signed_area(face, coords)) for face in faces]
    outer_idx = abs_areas.index(max(abs_areas))
    return [face for i, face in enumerate(faces) if i != outer_idx]


def vote_room_type(
    face: List[int],
    node_types: List[List[str]],
    all_nbrs: Dict[int, List[int]],
) -> str:
    face_set = set(face)
    face_counts: Counter = Counter()
    for node in face:
        for t in node_types[node]:
            face_counts[t] += 1

    if not face_counts:
        return "other"

    ext_counts: Counter = Counter()
    for node in face:
        for w in all_nbrs[node]:
            if w not in face_set:
                for t in node_types[w]:
                    ext_counts[t] += 1

    scores = {t: face_counts[t] / (ext_counts.get(t, 0) + 1) for t in face_counts}
    best_score = max(scores.values())
    winners = [t for t, s in scores.items() if s == best_score]
    order = {t: i for i, t in enumerate(ROOM_TYPE_ORDER)}
    return min(winners, key=lambda t: order.get(t, len(ROOM_TYPE_ORDER)))


def face_polygon(face: List[int], coords: List[Tuple[float, float]]) -> Optional[Polygon]:
    poly = Polygon([coords[i] for i in face])
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        return None
    if poly.geom_type != "Polygon":
        return None
    return poly


def is_axis_aligned_rectangle(poly: Polygon, tol: float = 1e-6) -> bool:
    minx, miny, maxx, maxy = poly.bounds
    env = box(minx, miny, maxx, maxy)
    return abs(poly.area - env.area) <= tol and poly.symmetric_difference(env).area <= tol


def make_sample_transform(polys: Sequence[Polygon], canvas: int = 256, pad: int = 16):
    union_poly = unary_union(list(polys))
    minx, miny, maxx, maxy = union_poly.bounds
    span_x = maxx - minx
    span_y = maxy - miny
    span = max(span_x, span_y, 1.0)
    target = canvas - 2 * pad
    scale = target / span

    def transform_xy(x: float, y: float) -> Tuple[int, int]:
        tx = int(round((x - minx) * scale + pad))
        ty = int(round((y - miny) * scale + pad))
        tx = max(0, min(canvas - 1, tx))
        ty = max(0, min(canvas - 1, ty))
        return tx, ty

    return transform_xy


def transform_polygon(poly: Polygon, transform_xy) -> Polygon:
    coords = [transform_xy(x, y) for x, y in poly.exterior.coords[:-1]]
    return Polygon(coords)


def room_location(poly: Polygon, plan_poly: Polygon) -> str:
    cx, cy = poly.centroid.x, poly.centroid.y
    minx, miny, maxx, maxy = plan_poly.bounds
    nx = (cx - minx) / max(maxx - minx, 1e-6)
    ny = (cy - miny) / max(maxy - miny, 1e-6)

    if nx < 1 / 3:
        xlab = "left"
    elif nx > 2 / 3:
        xlab = "right"
    else:
        xlab = "center"

    if ny < 1 / 3:
        ylab = "top"
    elif ny > 2 / 3:
        ylab = "bottom"
    else:
        ylab = "center"

    if xlab == "center" and ylab == "center":
        return "center"
    if xlab == "center":
        return ylab
    if ylab == "center":
        return xlab
    return f"{ylab}-{xlab}"


def rasterize_axis_aligned_boundary(poly: Polygon) -> List[List[int]]:
    pts = [(int(round(x)), int(round(y))) for x, y in poly.exterior.coords[:-1]]
    pixels: List[List[int]] = []
    for i, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(i + 1) % len(pts)]
        if x1 == x2:
            step = 1 if y2 >= y1 else -1
            for y in range(y1, y2 + step, step):
                pixels.append([x1, y])
        elif y1 == y2:
            step = 1 if x2 >= x1 else -1
            for x in range(x1, x2 + step, step):
                pixels.append([x, y1])
        else:
            raise ValueError("boundary is not axis-aligned")
    deduped: List[List[int]] = []
    seen = set()
    for p in pixels:
        key = (p[0], p[1])
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    return deduped


def make_boundary_boxes(plan_poly: Polygon) -> List[Dict[str, int]]:
    minx, miny, maxx, maxy = [int(round(v)) for v in plan_poly.bounds]
    cx = int(round((minx + maxx) / 2))
    cy = int(round((miny + maxy) / 2))
    h = maxx - minx
    w = maxy - miny
    return [{
        "room_type": "positive",
        "x": cx,
        "y": cy,
        "h": h,
        "w": w,
        "x_min": minx,
        "y_min": miny,
        "x_max": maxx,
        "y_max": maxy,
    }]


def tell2design_name_plan(face_types: List[str]) -> Optional[Dict[int, str]]:
    counts = Counter(face_types)
    if counts["living_room"] > 1 or counts["kitchen"] > 1 or counts["dining_room"] > 1:
        return None
    if counts["bathroom"] > 3:
        return None
    if counts["bedroom"] > 5:
        return None
    if max(counts["bedroom"] - 1, 0) + counts["corridor"] > 4:
        return None

    mapping: Dict[int, str] = {}
    bath_i = 1
    common_i = 1
    seen_master = False
    for idx, base in enumerate(face_types):
        if base == "living_room":
            mapping[idx] = "living room"
        elif base == "kitchen":
            mapping[idx] = "kitchen"
        elif base == "dining_room":
            mapping[idx] = "dining room"
        elif base == "bathroom":
            mapping[idx] = f"bathroom {bath_i}"
            bath_i += 1
        elif base == "bedroom":
            if not seen_master:
                mapping[idx] = "master room"
                seen_master = True
            else:
                if common_i > 4:
                    return None
                mapping[idx] = f"common room {common_i}"
                common_i += 1
        elif base == "corridor":
            if common_i > 4:
                return None
            mapping[idx] = f"common room {common_i}"
            common_i += 1
        else:
            return None
    return mapping


def build_relations(room_polys: Sequence[Polygon], room_names: Sequence[str]) -> Dict[str, List[str]]:
    relations: Dict[str, List[str]] = {name: [] for name in room_names}
    for i in range(len(room_polys)):
        for j in range(i + 1, len(room_polys)):
            inter = room_polys[i].boundary.intersection(room_polys[j].boundary)
            if inter.length > 0.5:
                relations[room_names[i]].append(room_names[j])
                relations[room_names[j]].append(room_names[i])
    return relations


def build_room_record(
    room_name: str,
    room_poly: Polygon,
    plan_poly: Polygon,
    relations: Dict[str, List[str]],
) -> Dict[str, object]:
    minx, miny, maxx, maxy = [int(round(v)) for v in room_poly.bounds]
    cx = int(round((minx + maxx) / 2))
    cy = int(round((miny + maxy) / 2))
    h = maxx - minx
    w = maxy - miny
    aspect = round(max(h, w) / max(min(h, w), 1), 3)
    return {
        "room_type": room_name,
        "x": cx,
        "y": cy,
        "h": h,
        "w": w,
        "x_min": minx,
        "y_min": miny,
        "x_max": maxx,
        "y_max": maxy,
        "near_x_min": minx,
        "near_y_min": miny,
        "near_x_max": maxx,
        "near_y_max": maxy,
        "relation": relations[room_name],
        "location": room_location(room_poly, plan_poly),
        "size": int(round(room_poly.area)),
        "aspect ratio": str(aspect),
        "private": None,
    }


def convert_sample(
    sample: Dict[str, object],
    id_to_combo: Dict[int, List[str]],
    drop_types: set,
) -> Tuple[Optional[Dict[str, object]], str]:
    n = int(sample["n_nodes"])
    raw_coords = sample["node_coords"][:n]
    adj = [row[:n] for row in sample["adj_matrix"][:n]]
    combo_ids = sample.get("node_combo_ids")
    if combo_ids is None:
        return None, "missing_node_combo_ids"

    coords = [(float(x), float(y)) for x, y in raw_coords]
    node_types = [id_to_combo.get(int(cid), ["other"]) for cid in combo_ids[:n]]
    all_nbrs = {i: [j for j in range(n) if i != j and adj[i][j] == 1] for i in range(n)}

    faces = find_faces(coords, adj)
    if not faces:
        return None, "no_faces"

    face_types: List[str] = []
    face_polys: List[Polygon] = []
    for face in faces:
        poly = face_polygon(face, coords)
        if poly is None:
            return None, "invalid_face_polygon"
        face_type = vote_room_type(face, node_types, all_nbrs)
        if face_type in drop_types or face_type not in SUPPORTED_BASE_TYPES:
            return None, f"unsupported_type:{face_type}"
        if not is_axis_aligned_rectangle(poly):
            return None, "non_rect_room"
        face_types.append(face_type)
        face_polys.append(poly)

    name_map = tell2design_name_plan(face_types)
    if name_map is None:
        return None, "unsupported_room_count"

    transform_xy = make_sample_transform(face_polys)
    tpolys = [transform_polygon(poly, transform_xy) for poly in face_polys]
    if not all(is_axis_aligned_rectangle(poly) for poly in tpolys):
        return None, "transform_non_rect_room"

    plan_poly = unary_union(tpolys)
    if plan_poly.geom_type != "Polygon":
        return None, "non_single_plan_polygon"

    room_names = [name_map[i] for i in range(len(tpolys))]
    relations = build_relations(tpolys, room_names)
    rooms = [
        build_room_record(room_names[i], tpolys[i], plan_poly, relations)
        for i in range(len(tpolys))
    ]

    image = str(sample.get("image", ""))
    img_id = Path(image).stem if image else f"{sample.get('source_file', 'sample')}_{sample.get('source_line', 0)}"
    out = {
        "annotated_strings": sample.get("prompt_original") or sample.get("prompt") or "",
        "boundary": rasterize_axis_aligned_boundary(plan_poly),
        "boundary_boxs": make_boundary_boxes(plan_poly),
        "rooms": rooms,
        "img_id": img_id,
    }
    return out, "ok"


def init_worker(id_to_combo: Dict[int, List[str]], drop_types: set):
    global WORKER_ID_TO_COMBO, WORKER_DROP_TYPES
    WORKER_ID_TO_COMBO = id_to_combo
    WORKER_DROP_TYPES = drop_types


def convert_sample_worker(task: Tuple[int, Dict[str, object]]) -> Tuple[int, Optional[Dict[str, object]], str]:
    raw_idx, sample = task
    converted, reason = convert_sample(sample, WORKER_ID_TO_COMBO, WORKER_DROP_TYPES)
    return raw_idx, converted, reason


def parse_args():
    p = argparse.ArgumentParser(description="Convert graph JSONL to Tell2Design floorplan JSON.")
    p.add_argument(
        "--split",
        choices=["train", "val", "all"],
        default="train",
        help="Which default input/output pair to use when --input/--output are omitted.",
    )
    p.add_argument(
        "--input",
        default=None,
        help="Input graph jsonl. Defaults are centralized in the script by --split.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output floorplan json. Defaults to Tell2Design-comp/T5/data/floorplan/floorplan_train.json or floorplan_dev.json",
    )
    p.add_argument(
        "--combo-vocab",
        default=DEFAULT_COMBO_VOCAB,
        help="Combo vocab used to decode node_combo_ids.",
    )
    p.add_argument(
        "--drop-types",
        nargs="*",
        default=["other"],
        help="Base room types that make a sample ineligible.",
    )
    p.add_argument(
        "--stats-out",
        default=None,
        help="Optional stats json path. Defaults next to output json with _stats suffix.",
    )
    p.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N raw jsonl samples before conversion.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert at most N raw jsonl samples after --offset. Useful for quick validation.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of worker processes for parallel conversion. Use 1 to disable multiprocessing.",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=64,
        help="Task chunksize passed to the process pool.",
    )
    return p.parse_args()


def resolve_default_input(split: str) -> Path:
    if split == "train":
        return Path(DEFAULT_TRAIN_INPUT)
    if split == "val":
        return Path(DEFAULT_VAL_INPUT)
    return Path(DEFAULT_ALL_INPUT)


def resolve_default_output(split: str) -> Path:
    output_dir = Path(DEFAULT_OUTPUT_DIR)
    if split == "train":
        return output_dir / "floorplan_train.json"
    if split == "val":
        return output_dir / "floorplan_dev.json"
    return output_dir / "floorplan_all.json"


def main():
    args = parse_args()
    input_path = Path(args.input) if args.input is not None else resolve_default_input(args.split)
    if args.output is None:
        output_path = resolve_default_output(args.split)
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    id_to_combo = load_vocab(Path(args.combo_vocab))
    drop_types = set(args.drop_types)

    tasks = []
    for raw_idx, (_, sample) in enumerate(load_jsonl(input_path)):
        if raw_idx < args.offset:
            continue
        if args.limit is not None and len(tasks) >= args.limit:
            break
        tasks.append((raw_idx, sample))

    results = []
    stats = Counter()
    if args.workers <= 1:
        init_worker(id_to_combo, drop_types)
        iterator = (
            convert_sample_worker(task)
            for task in tqdm(tasks, desc="convert", unit="sample")
        )
    else:
        pool = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(id_to_combo, drop_types),
        )
        iterator = pool.map(convert_sample_worker, tasks, chunksize=args.chunksize)
        iterator = tqdm(iterator, total=len(tasks), desc="convert", unit="sample")

    ordered_results: List[Tuple[int, Dict[str, object]]] = []
    try:
        for raw_idx, converted, reason in iterator:
            stats["total"] += 1
            stats[reason] += 1
            if converted is not None:
                ordered_results.append((raw_idx, converted))
    finally:
        if args.workers > 1:
            pool.shutdown(wait=True)

    ordered_results.sort(key=lambda item: item[0])
    results = [converted for _, converted in ordered_results]

    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "offset": args.offset,
        "limit": args.limit,
        "workers": args.workers,
        "total_samples": stats["total"],
        "kept_samples": len(results),
        "drop_samples": stats["total"] - len(results),
        "reasons": dict(sorted((k, int(v)) for k, v in stats.items() if k != "total")),
    }

    if args.stats_out:
        stats_path = Path(args.stats_out)
    else:
        stats_path = output_path.with_name(f"{output_path.stem}_stats.json")

    if stats_path:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
