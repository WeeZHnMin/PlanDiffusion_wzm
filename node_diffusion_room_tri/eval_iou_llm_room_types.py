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
import importlib.util
import json
import math
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .graph_prune import prune_dangling_nodes

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
MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
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
    p.add_argument("--provider", choices=["auto", "dashscope", "openai", "mimo"], default="auto")
    p.add_argument("--base-url", "--base_url", dest="base_url",
                   default=os.environ.get("OPENAI_BASE_URL"))
    p.add_argument("--api-key", "--api_key", dest="api_key",
                   default=os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("MIMO_API_KEY"))
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=0.95)
    p.add_argument("--max-completion-tokens", "--max_completion_tokens",
                   dest="max_completion_tokens", type=int, default=None,
                   help="Optional. Default: do not limit completion tokens.")
    p.add_argument("--frequency-penalty", "--frequency_penalty",
                   dest="frequency_penalty", type=float, default=0.0)
    p.add_argument("--presence-penalty", "--presence_penalty",
                   dest="presence_penalty", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--retry-times", type=int, default=3)
    p.add_argument("--retry-delay", type=float, default=1.0)
    p.add_argument("--disable-thinking", action="store_true",
                   help="Disable model thinking mode")
    p.add_argument("--enable-thinking", action="store_true", default=True,
                   help="Enable model thinking mode (default: on)")
    p.add_argument("--workers", type=int, default=16)
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


def center_at_origin(coords: np.ndarray) -> np.ndarray:
    if len(coords) == 0:
        return coords
    return coords - coords.mean(axis=0, keepdims=True)


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
    coords_np = np.array(coords_raw[:n], dtype=np.float32)
    adj = np.array(adj_raw, dtype=np.int32)[:n, :n]
    np.fill_diagonal(adj, 0)
    node_types_list = [
        (t if isinstance(t, list) else [t])
        for t in (node_types[:n] if node_types is not None else [])
    ]
    if len(node_types_list) < n:
        node_types_list.extend([["bedroom"]] * (n - len(node_types_list)))
    coords_np, adj, node_types_list, _ = prune_dangling_nodes(
        coords_np, adj, node_types_list
    )
    n = len(coords_np)
    if n < 3:
        return {}
    coords_np = center_at_origin(coords_np)
    coords = [(float(coords_np[i, 0]), float(coords_np[i, 1])) for i in range(n)]
    faces = find_faces(coords, adj)
    all_nbrs = _build_sorted_neighbors(coords, adj, n)
    out: Dict[str, List[Polygon]] = {}
    for face in faces:
        rtype = vote_gt_room_type(face, node_types_list, all_nbrs)
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
    coords_np = np.array(row["pred_node_coords"][:n], dtype=np.float32)
    coords_np = center_at_origin(coords_np)
    coords = [(float(coords_np[i, 0]), float(coords_np[i, 1])) for i in range(n)]
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
        "You are given a generated floor-plan graph and a natural-language description of the intended layout.\n"
        "Each recovered room includes its nodes, node coordinates, center, bounding box, coarse location, and adjacent rooms.\n"
        "Use all of this information together with the text_prompt to assign the most plausible semantic type to every recovered room.\n"
        "Choose the room types that make the generated rooms best match the text description, including relative positions and adjacency relationships.\n"
        "Use only the allowed room types. Do not use or infer from any ground-truth fields.\n"
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
    provider: str,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_completion_tokens: Optional[int],
    frequency_penalty: float,
    presence_penalty: float,
    disable_thinking: bool,
    retry_times: int,
    retry_delay: float,
) -> str:
    last_error = ""
    for attempt in range(1, retry_times + 1):
        try:
            kwargs = {}
            if provider == "mimo":
                kwargs.update({
                    "top_p": top_p,
                    "stream": False,
                    "stop": None,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                })
                if max_completion_tokens is not None:
                    kwargs["max_completion_tokens"] = max_completion_tokens
            else:
                kwargs["extra_body"] = {"enable_thinking": not disable_thinking}
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


def base_output(row: Dict[str, Any], row_index: int) -> Dict[str, Any]:
    out = dict(row)
    out.update({
        "row_index": row_index,
        "source_index": row.get("source_index", row_index),
        "status": "",
        "error": "",
        "retry_count": 0,
        "n_recovered_rooms": 0,
        "recovered_rooms": [],
        "room_types": [],
        "micro_iou": None,
        "macro_iou": None,
        "raw_response_1": None,
        "raw_response_2": None,
    })
    return out


def process_one(
    row_index: int,
    row: Dict[str, Any],
    args: argparse.Namespace,
    clients: Sequence[Any],
    provider: str,
    disable_thinking: bool,
) -> Dict[str, Any]:
    out = base_output(row, row_index)
    raw1 = None
    raw2 = None
    try:
        rooms, pred_polys_by_room = recover_pred_rooms(row)
        out["n_recovered_rooms"] = len(rooms)
        out["recovered_rooms"] = rooms
        if not rooms:
            raise ValueError("no recovered generated rooms")

        prompt = build_prompt(row, rooms)
        if args.dry_run:
            out["status"] = "dry_run"
            out["llm_prompt"] = prompt
            return out

        expected_ids = [r["room_id"] for r in rooms]
        client = clients[row_index % len(clients)]
        raw1 = call_llm(
            client, provider, args.model, prompt, args.temperature,
            args.top_p, args.max_completion_tokens,
            args.frequency_penalty, args.presence_penalty,
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
                client, provider, args.model, retry_prompt, args.temperature,
                args.top_p, args.max_completion_tokens,
                args.frequency_penalty, args.presence_penalty,
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
        out.update({
            "status": "ok",
            "retry_count": retry_count,
            "room_types": [{"room_id": rid, "type": room_types[rid]} for rid in expected_ids],
            "micro_iou": round(micro, 6),
            "macro_iou": round(macro, 6),
            "raw_response_1": raw1,
            "raw_response_2": raw2,
        })
        if args.sleep > 0:
            time.sleep(args.sleep)
        return out
    except Exception as e:
        out.update({
            "status": "parse_failed" if raw1 is not None else "invalid",
            "error": str(e),
            "raw_response_1": raw1,
            "raw_response_2": raw2,
        })
        if args.strict:
            raise
        return out


def resolve_provider(args: argparse.Namespace) -> str:
    if args.provider != "auto":
        return args.provider
    if args.model.lower().startswith("mimo") or (args.base_url and "xiaomimimo" in args.base_url):
        return "mimo"
    if args.base_url:
        return "openai"
    return "dashscope"


def load_mimo_keys() -> List[str]:
    keys: List[str] = []
    env_key = os.environ.get("MIMO_API_KEY")
    if env_key:
        keys.append(env_key)
    key_file = Path("api_keys.py")
    if key_file.exists():
        spec = importlib.util.spec_from_file_location("local_api_keys", key_file)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            keys.extend([str(k) for k in getattr(mod, "MIMO_API_KEYS", []) if k])
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(keys))


def make_clients(args: argparse.Namespace, provider: str) -> Tuple[List[Any], str]:
    base_url = args.base_url
    if provider == "mimo":
        base_url = base_url or MIMO_BASE_URL
        keys = [args.api_key] if args.api_key else load_mimo_keys()
    elif provider == "dashscope":
        base_url = base_url or DEFAULT_BASE_URL
        keys = [args.api_key] if args.api_key else []
    else:
        keys = [args.api_key] if args.api_key else []
    keys = [k for k in keys if k]
    if not keys:
        raise RuntimeError(f"API key is required for provider={provider}")
    return [OpenAI(api_key=k, base_url=base_url, timeout=args.timeout) for k in keys], base_url or ""


def main() -> None:
    args = parse_args()
    provider = resolve_provider(args)
    if not args.dry_run:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        clients, base_url = make_clients(args, provider)
        print(f"provider={provider} model={args.model} base_url={base_url} clients={len(clients)}")
    else:
        clients = []
    disable_thinking = args.disable_thinking

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary) if args.summary else out_path.with_suffix(".summary.json")

    rows: List[Tuple[int, Dict[str, Any]]] = []
    for idx, row in enumerate(read_jsonl(args.jsonl)):
        if args.n_samples > 0 and len(rows) >= args.n_samples:
            break
        rows.append((idx, row))

    results: Dict[int, Dict[str, Any]] = {}
    workers = max(1, int(args.workers))
    if workers == 1:
        for idx, row in rows:
            results[idx] = process_one(idx, row, args, clients, provider, disable_thinking)
            print(f"[{len(results)}/{len(rows)}] row={idx} status={results[idx]['status']}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(process_one, idx, row, args, clients, provider, disable_thinking): idx
                for idx, row in rows
            }
            for fut in as_completed(futs):
                idx = futs[fut]
                results[idx] = fut.result()
                print(f"[{len(results)}/{len(rows)}] row={idx} status={results[idx]['status']}", flush=True)

    with out_path.open("w", encoding="utf-8") as fout:
        for idx, _ in rows:
            fout.write(json.dumps(results[idx], ensure_ascii=False) + "\n")

    ok_rows = [r for r in results.values() if r["status"] == "ok"]
    micro_list = [float(r["micro_iou"]) for r in ok_rows if r["micro_iou"] is not None]
    macro_list = [float(r["macro_iou"]) for r in ok_rows if r["macro_iou"] is not None]

    summary = {
        "total": len(rows),
        "ok": sum(1 for r in results.values() if r["status"] == "ok"),
        "parse_failed": sum(1 for r in results.values() if r["status"] == "parse_failed"),
        "invalid": sum(1 for r in results.values() if r["status"] == "invalid"),
        "dry_run": sum(1 for r in results.values() if r["status"] == "dry_run"),
        "workers": workers,
        "micro_iou": float(np.mean(micro_list)) if micro_list else 0.0,
        "macro_iou": float(np.mean(macro_list)) if macro_list else 0.0,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"details -> {out_path}")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
