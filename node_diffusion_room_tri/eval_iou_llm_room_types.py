"""Evaluate generated tri layouts with LLM-assigned room types.

Input is an inference JSONL such as outputs/tri_from_llm_graph_ddim500.jsonl
containing:
  prompt, adj_matrix, pred_node_coords,
  gt_adj_matrix, gt_node_coords, gt_node_types

For each sample, the script recovers generated room faces from
adj_matrix + pred_node_coords, asks an LLM to assign one room type per face,
parses the JSON answer with one retry, then computes IoU against GT rooms.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


ALLOWED_TYPES = {
    "bathroom",
    "bedroom",
    "living_room",
    "kitchen",
    "corridor",
    "dining_room",
}

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.5-plus"

TYPE_ALIASES = {
    "bath": "bathroom",
    "toilet": "bathroom",
    "wc": "bathroom",
    "master room": "bedroom",
    "master_room": "bedroom",
    "common room": "bedroom",
    "common_room": "bedroom",
    "bed room": "bedroom",
    "living": "living_room",
    "living room": "living_room",
    "livingroom": "living_room",
    "hall": "living_room",
    "cook": "kitchen",
    "corridor room": "corridor",
    "hallway": "corridor",
    "dining": "dining_room",
    "dining room": "dining_room",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True, help="tri inference output JSONL")
    p.add_argument("--out", default="outputs/tri_llm_room_type_iou.jsonl")
    p.add_argument("--summary", default=None, help="Optional summary JSON path")
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    p.add_argument("--base-url", "--base_url", dest="base_url",
                   default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--api-key", "--api_key", dest="api_key",
                   default=os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--retry-times", type=int, default=3)
    p.add_argument("--retry-delay", type=float, default=1.0)
    p.add_argument("--disable-thinking", action="store_true",
                   help="Disable model thinking mode")
    p.add_argument("--enable-thinking", action="store_true", default=True,
                   help="Enable model thinking mode (default: on)")
    p.add_argument("--n_samples", type=int, default=0, help="0 means all")
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--dry_run", action="store_true", help="Build prompts only; do not call LLM")
    p.add_argument("--strict", action="store_true", help="Fail on missing required fields")
    return p.parse_args()


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_sorted_neighbors(coords: Sequence[Tuple[float, float]], adj: np.ndarray, n: int):
    nbrs = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] == 1:
                nbrs[i].append(j)
    for i in range(n):
        nbrs[i] = sorted(
            nbrs[i],
            key=lambda w: math.atan2(coords[w][1] - coords[i][1], coords[w][0] - coords[i][0]),
        )
    return nbrs


def _next_half_edge(u: int, v: int, sorted_nbrs: Dict[int, List[int]]) -> Optional[int]:
    nbrs = sorted_nbrs[v]
    if not nbrs:
        return None
    return nbrs[(nbrs.index(u) - 1) % len(nbrs)]


def _signed_area(face: Sequence[int], coords: Sequence[Tuple[float, float]]) -> float:
    pts = [coords[i] for i in face]
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def find_faces(coords: Sequence[Tuple[float, float]], adj: np.ndarray) -> List[List[int]]:
    n = len(coords)
    sorted_nbrs = _build_sorted_neighbors(coords, adj, n)
    visited, faces = set(), []
    for u in range(n):
        for v in sorted_nbrs[u]:
            if (u, v) in visited:
                continue
            face, cu, cv, steps = [], u, v, 0
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
    abs_areas = [abs(_signed_area(f, coords)) for f in faces]
    outer_idx = abs_areas.index(max(abs_areas))
    return [f for i, f in enumerate(faces) if i != outer_idx]


def norm_type(value: Any) -> str:
    s = str(value).strip().lower().replace("-", "_")
    s = re.sub(r"\s+", " ", s)
    s = TYPE_ALIASES.get(s, s)
    s = s.replace(" ", "_")
    if s not in ALLOWED_TYPES:
        raise ValueError(f"invalid room type: {value!r}")
    return s


def vote_gt_room_type(face: Sequence[int], node_types: Sequence[Any], all_nbrs: Dict[int, List[int]]) -> str:
    face_set = set(face)
    face_counts = Counter()
    for node in face:
        vals = node_types[node] if isinstance(node_types[node], list) else [node_types[node]]
        for t in vals:
            try:
                face_counts[norm_type(t)] += 1
            except ValueError:
                pass
    if not face_counts:
        return "bedroom"
    ext_counts = Counter()
    for node in face:
        for w in all_nbrs[node]:
            if w not in face_set:
                vals = node_types[w] if isinstance(node_types[w], list) else [node_types[w]]
                for t in vals:
                    try:
                        ext_counts[norm_type(t)] += 1
                    except ValueError:
                        pass
    scores = {t: face_counts[t] / (ext_counts.get(t, 0) + 1) for t in face_counts}
    best = max(scores.values())
    winners = [t for t, s in scores.items() if s == best]
    order = {t: i for i, t in enumerate(["bathroom", "bedroom", "living_room", "kitchen", "corridor", "dining_room"])}
    return min(winners, key=lambda t: order.get(t, 999))


def polys_by_type_from_gt(coords_raw: Any, adj_raw: Any, node_types: Any) -> Dict[str, List[Polygon]]:
    n = min(len(coords_raw), len(adj_raw))
    coords = [(float(coords_raw[i][0]), float(coords_raw[i][1])) for i in range(n)]
    adj = np.array(adj_raw, dtype=np.int32)[:n, :n]
    np.fill_diagonal(adj, 0)
    faces = find_faces(coords, adj)
    all_nbrs = _build_sorted_neighbors(coords, adj, n)
    out: Dict[str, List[Polygon]] = {}
    for face in faces:
        rtype = vote_gt_room_type(face, node_types, all_nbrs)
        poly = safe_poly([coords[i] for i in face])
        if poly is not None:
            out.setdefault(rtype, []).append(poly)
    return out


def safe_poly(points: Sequence[Tuple[float, float]]) -> Optional[Polygon]:
    try:
        poly = Polygon(points)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_valid and poly.area > 1e-6:
            return poly
    except Exception:
        pass
    return None


def room_location(cx: float, cy: float, min_x: float, max_x: float, min_y: float, max_y: float) -> str:
    xmid = (min_x + max_x) / 2.0
    ymid = (min_y + max_y) / 2.0
    ew = "west" if cx < xmid else "east"
    ns = "south" if cy < ymid else "north"
    if abs(cx - xmid) < (max_x - min_x) * 0.12:
        ew = "center"
    if abs(cy - ymid) < (max_y - min_y) * 0.12:
        ns = "middle"
    if ew == "center" and ns == "middle":
        return "center"
    if ew == "center":
        return ns
    if ns == "middle":
        return ew
    return f"{ns}-{ew}"


def recover_pred_rooms(row: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, List[Polygon]]]:
    n = int(row["n_nodes"])
    coords = [(float(x), float(y)) for x, y in row["pred_node_coords"][:n]]
    adj = np.array(row["adj_matrix"], dtype=np.int32)[:n, :n]
    np.fill_diagonal(adj, 0)
    faces = find_faces(coords, adj)
    if not faces:
        return [], {}
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)

    face_sets = [set(f) for f in faces]
    rooms = []
    polys_by_room = {}
    for idx, face in enumerate(faces, start=1):
        pts = [coords[i] for i in face]
        poly = safe_poly(pts)
        if poly is None:
            continue
        rid = f"R{idx}"
        cx, cy = float(poly.centroid.x), float(poly.centroid.y)
        bx0, by0, bx1, by1 = poly.bounds
        adjacent = []
        for j, other in enumerate(face_sets, start=1):
            if j == idx:
                continue
            if len(set(face) & other) >= 2:
                adjacent.append(f"R{j}")
        rooms.append({
            "room_id": rid,
            "nodes": list(map(int, face)),
            "coordinates": {str(i): [round(coords[i][0], 3), round(coords[i][1], 3)] for i in face},
            "center": [round(cx, 3), round(cy, 3)],
            "bbox": [round(bx0, 3), round(by0, 3), round(bx1, 3), round(by1, 3)],
            "location": room_location(cx, cy, min_x, max_x, min_y, max_y),
            "adjacent_rooms": adjacent,
        })
        polys_by_room[rid] = [poly]
    return rooms, polys_by_room


def build_prompt(row: Dict[str, Any], rooms: List[Dict[str, Any]]) -> str:
    payload = {
        "coordinate_system": "x increases east/right; y increases north/up",
        "allowed_room_types": sorted(ALLOWED_TYPES),
        "text_prompt": row.get("prompt", ""),
        "rooms": rooms,
    }
    return (
        "Assign a semantic room type to each recovered room in the generated floor plan.\n"
        "Use only the allowed room types. Do not use ground-truth information.\n"
        "Return only strict JSON in this exact schema:\n"
        '{"rooms":[{"room_id":"R1","type":"bedroom"}]}\n\n'
        f"Input:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def parse_llm_response(text: str, expected_room_ids: Sequence[str]) -> Dict[str, str]:
    obj = extract_json(text)
    if not isinstance(obj, dict) or not isinstance(obj.get("rooms"), list):
        raise ValueError("response must be a JSON object with rooms list")
    parsed: Dict[str, str] = {}
    for item in obj["rooms"]:
        if not isinstance(item, dict):
            raise ValueError("each room item must be an object")
        rid = str(item.get("room_id", "")).strip()
        if rid not in expected_room_ids:
            raise ValueError(f"unknown room_id: {rid}")
        if rid in parsed:
            raise ValueError(f"duplicate room_id: {rid}")
        parsed[rid] = norm_type(item.get("type"))
    missing = [rid for rid in expected_room_ids if rid not in parsed]
    if missing:
        raise ValueError(f"missing room ids: {missing}")
    return parsed


def call_llm(
    client: Any,
    model: str,
    prompt: str,
    temperature: float,
    disable_thinking: bool,
    retry_times: int,
    retry_delay: float,
) -> str:
    last_error = ""
    for attempt in range(1, retry_times + 1):
        try:
            kwargs = {}
            extra_body = {}
            if disable_thinking:
                extra_body["enable_thinking"] = False
            else:
                extra_body["enable_thinking"] = True
            if extra_body:
                kwargs["extra_body"] = extra_body
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You label floor-plan rooms. Output strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                **kwargs,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_error = str(e)
            if attempt < retry_times:
                time.sleep(retry_delay * attempt)
    raise RuntimeError(last_error)


def compute_iou(gt_by_type: Dict[str, List[Polygon]], pred_by_type: Dict[str, List[Polygon]]) -> Tuple[float, float]:
    all_types = set(gt_by_type) | set(pred_by_type)
    intersections, unions, ious = [], [], []
    for rtype in all_types:
        gt_u = unary_union(gt_by_type.get(rtype, []) or [Polygon()])
        pred_u = unary_union(pred_by_type.get(rtype, []) or [Polygon()])
        inter = gt_u.intersection(pred_u).area
        union = gt_u.union(pred_u).area
        if union > 1e-9:
            intersections.append(inter)
            unions.append(union)
            ious.append(inter / union)
    if not unions:
        return 0.0, 0.0
    return float(sum(intersections) / sum(unions)), float(sum(ious) / len(ious))


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        if not args.api_key:
            raise RuntimeError("OPENAI_API_KEY or --api_key is required")
        client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)
    else:
        client = None
    disable_thinking = args.disable_thinking

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary) if args.summary else out_path.with_suffix(".summary.json")

    total = ok = parse_failed = invalid = dry = 0
    micro_list: List[float] = []
    macro_list: List[float] = []

    with out_path.open("w", encoding="utf-8") as fout:
        for idx, row in enumerate(read_jsonl(args.jsonl)):
            if args.n_samples > 0 and total >= args.n_samples:
                break
            total += 1
            source_index = row.get("source_index", idx)
            raw1 = None
            raw2 = None
            try:
                rooms, pred_polys_by_room = recover_pred_rooms(row)
                if not rooms:
                    raise ValueError("no recovered generated rooms")
                prompt = build_prompt(row, rooms)
                if args.dry_run:
                    fout.write(json.dumps({
                        "source_index": source_index,
                        "status": "dry_run",
                        "prompt": prompt,
                        "rooms": rooms,
                    }, ensure_ascii=False) + "\n")
                    dry += 1
                    continue

                expected_ids = [r["room_id"] for r in rooms]
                raw1 = call_llm(
                    client, args.model, prompt, args.temperature,
                    disable_thinking, args.retry_times, args.retry_delay,
                )
                try:
                    room_types = parse_llm_response(raw1, expected_ids)
                    retry_count = 0
                except Exception as e1:
                    retry_prompt = (
                        f"{prompt}\n\nYour previous answer could not be parsed: {e1}.\n"
                        "Return only strict JSON. Cover every room_id exactly once."
                    )
                    raw2 = call_llm(
                        client, args.model, retry_prompt, args.temperature,
                        disable_thinking, args.retry_times, args.retry_delay,
                    )
                    room_types = parse_llm_response(raw2, expected_ids)
                    retry_count = 1

                pred_by_type: Dict[str, List[Polygon]] = {}
                for rid, rtype in room_types.items():
                    pred_by_type.setdefault(rtype, []).extend(pred_polys_by_room[rid])

                gt_by_type = polys_by_type_from_gt(
                    row["gt_node_coords"],
                    row["gt_adj_matrix"],
                    row["gt_node_types"],
                )
                micro, macro = compute_iou(gt_by_type, pred_by_type)
                micro_list.append(micro)
                macro_list.append(macro)
                ok += 1
                fout.write(json.dumps({
                    "source_index": source_index,
                    "status": "ok",
                    "retry_count": retry_count,
                    "room_types": [{"room_id": rid, "type": room_types[rid]} for rid in expected_ids],
                    "micro_iou": round(micro, 6),
                    "macro_iou": round(macro, 6),
                    "raw_response_1": raw1,
                    "raw_response_2": raw2,
                }, ensure_ascii=False) + "\n")
                if args.sleep > 0:
                    time.sleep(args.sleep)
            except Exception as e:
                status = "parse_failed" if raw1 is not None else "invalid"
                if status == "parse_failed":
                    parse_failed += 1
                else:
                    invalid += 1
                fout.write(json.dumps({
                    "source_index": source_index,
                    "status": status,
                    "error": str(e),
                    "raw_response_1": raw1,
                    "raw_response_2": raw2,
                }, ensure_ascii=False) + "\n")
                if args.strict:
                    raise

    summary = {
        "total": total,
        "ok": ok,
        "parse_failed": parse_failed,
        "invalid": invalid,
        "dry_run": dry,
        "micro_iou": float(np.mean(micro_list)) if micro_list else 0.0,
        "macro_iou": float(np.mean(macro_list)) if macro_list else 0.0,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"details -> {out_path}")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
