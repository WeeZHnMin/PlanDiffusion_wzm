import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert old-combo dataset files to parquet and upload them to a Hugging Face dataset repo."
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=Path("data/processed/graph_tokens_combo_from_final_old.npz"),
        help="Path to the NPZ file containing token/graph arrays.",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("data/processed/graph_prompts_combo_from_final_old.txt"),
        help="Path to the prompt txt file aligned with the NPZ rows.",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path("data/processed/type_combo_vocab_old.json"),
        help="Path to the old combo vocab json.",
    )
    parser.add_argument(
        "--jsonl-prompts",
        type=Path,
        default=Path("data/jsonl/final_graph_dataset.jsonl"),
        help="Fallback jsonl source for prompts when the txt file contains embedded newlines.",
    )
    parser.add_argument(
        "--repo-id",
        default="wzmmmm/plan-diffusion",
        help="Target Hugging Face dataset repo.",
    )
    parser.add_argument(
        "--repo-type",
        default="dataset",
        help="Hugging Face repo type.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split name used in parquet shard filenames.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=9,
        help="Number of parquet shards to create.",
    )
    parser.add_argument(
        "--commit-message",
        default="Replace parquet dataset with old-combo reencoded data",
        help="Commit message for the HF upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build parquet files locally without uploading.",
    )
    return parser.parse_args()


def load_prompts(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def load_prompts_from_jsonl(path: Path, expected_total: int):
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["prompt"])

    if not prompts:
        raise ValueError(f"No prompts found in fallback jsonl: {path}")
    if expected_total % len(prompts) != 0:
        raise ValueError(
            f"NPZ rows ({expected_total}) are not divisible by fallback prompt rows ({len(prompts)})."
        )

    repeat = expected_total // len(prompts)
    rebuilt = []
    for prompt in prompts:
        rebuilt.extend([prompt] * repeat)
    return rebuilt, repeat


def shard_bounds(total, num_shards):
    shard_size = math.ceil(total / num_shards)
    for shard_idx in range(num_shards):
        start = shard_idx * shard_size
        end = min(total, start + shard_size)
        if start >= end:
            break
        yield shard_idx, start, end


def rows_to_table(npz_data, prompts, start, end):
    tokens = npz_data["tokens"]
    lengths = npz_data["lengths"]
    coords = npz_data["coords"]
    node_types = npz_data["node_types"]
    adj_matrix = npz_data["adj_matrix"]
    n_nodes = npz_data["n_nodes"]

    shard = {
        "tokens": [],
        "lengths": [],
        "coords": [],
        "node_types": [],
        "adj_matrix": [],
        "n_nodes": [],
        "prompt": [],
    }

    for idx in range(start, end):
        length = int(lengths[idx])
        shard["tokens"].append(tokens[idx, :length].astype(np.int64).tolist())
        shard["lengths"].append(length)
        shard["coords"].append(coords[idx].astype(np.float64).tolist())
        shard["node_types"].append(node_types[idx].astype(np.int64).tolist())
        shard["adj_matrix"].append(adj_matrix[idx].astype(np.float64).tolist())
        shard["n_nodes"].append(int(n_nodes[idx]))
        shard["prompt"].append(prompts[idx])

    return pa.table(shard)


def build_vocab_table(vocab_obj):
    row = {
        "combo_to_id_json": [json.dumps(vocab_obj.get("combo_to_id", {}), ensure_ascii=False)],
        "base_type_names_json": [json.dumps(vocab_obj.get("base_type_names", {}), ensure_ascii=False)],
        "N_TYPES": [int(vocab_obj["N_TYPES"])],
        "TOK_OPEN": [int(vocab_obj["TOK_OPEN"])],
        "TOK_CLOSE": [int(vocab_obj["TOK_CLOSE"])],
        "TOK_BREAK": [int(vocab_obj["TOK_BREAK"])],
        "BOS_ID": [int(vocab_obj["BOS_ID"])],
        "EOS_ID": [int(vocab_obj["EOS_ID"])],
        "NODE_OFFSET": [int(vocab_obj["NODE_OFFSET"])],
        "VOCAB_SIZE": [int(vocab_obj["VOCAB_SIZE"])],
        "MAX_NODES": [int(vocab_obj["MAX_NODES"])],
    }
    return pa.table(row)


def write_parquet_files(tmpdir: Path, npz_data, prompts, vocab_obj, split, num_shards):
    data_dir = tmpdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    total = len(prompts)
    shard_paths = []
    for shard_idx, start, end in shard_bounds(total, num_shards):
        shard_name = f"{split}-{shard_idx:05d}-of-{num_shards:05d}.parquet"
        shard_path = data_dir / shard_name
        table = rows_to_table(npz_data, prompts, start, end)
        pq.write_table(table, shard_path, compression="snappy")
        shard_paths.append(shard_path)
        print(f"built {shard_name}  rows={end - start}")

    vocab_parquet_path = data_dir / "type_combo_vocab_old.parquet"
    pq.write_table(build_vocab_table(vocab_obj), vocab_parquet_path, compression="snappy")
    print(f"built {vocab_parquet_path.name}")

    vocab_json_path = tmpdir / "type_combo_vocab.json"
    vocab_json_path.write_text(
        json.dumps(vocab_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"built {vocab_json_path.name}")

    return shard_paths, vocab_parquet_path, vocab_json_path


def upload_to_hf(repo_id, repo_type, commit_message, local_shards, vocab_parquet_path, vocab_json_path):
    api = HfApi()
    existing_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)

    operations = []
    for repo_file in existing_files:
        if repo_file.startswith("data/train-") and repo_file.endswith(".parquet"):
            operations.append(CommitOperationDelete(path_in_repo=repo_file))
        if repo_file == "data/type_combo_vocab_old.parquet":
            operations.append(CommitOperationDelete(path_in_repo=repo_file))

    for shard_path in local_shards:
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"data/{shard_path.name}",
                path_or_fileobj=str(shard_path),
            )
        )

    operations.append(
        CommitOperationAdd(
            path_in_repo="data/type_combo_vocab_old.parquet",
            path_or_fileobj=str(vocab_parquet_path),
        )
    )
    operations.append(
        CommitOperationAdd(
            path_in_repo="type_combo_vocab.json",
            path_or_fileobj=str(vocab_json_path),
        )
    )

    result = api.create_commit(
        repo_id=repo_id,
        repo_type=repo_type,
        operations=operations,
        commit_message=commit_message,
    )
    return result


def main():
    args = parse_args()

    npz_data = np.load(args.npz, allow_pickle=False)
    vocab_obj = json.loads(args.vocab.read_text(encoding="utf-8"))

    expected_total = int(npz_data["tokens"].shape[0])
    prompts = load_prompts(args.prompts)
    if len(prompts) != expected_total:
        prompts, repeat = load_prompts_from_jsonl(args.jsonl_prompts, expected_total)
        print(
            f"prompt txt line count mismatch ({len(load_prompts(args.prompts))} vs {expected_total}); "
            f"rebuilt prompts from {args.jsonl_prompts} with repeat={repeat}"
        )

    total = len(prompts)

    print(f"rows: {total}")
    print(f"num_shards: {args.num_shards}")
    print(f"repo: {args.repo_id}")

    with tempfile.TemporaryDirectory(prefix="hf_old_combo_") as tmp:
        tmpdir = Path(tmp)
        local_shards, vocab_parquet_path, vocab_json_path = write_parquet_files(
            tmpdir,
            npz_data,
            prompts,
            vocab_obj,
            args.split,
            args.num_shards,
        )

        if args.dry_run:
            print(f"dry run complete -> {tmpdir}")
            return

        result = upload_to_hf(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            commit_message=args.commit_message,
            local_shards=local_shards,
            vocab_parquet_path=vocab_parquet_path,
            vocab_json_path=vocab_json_path,
        )
        print(f"uploaded -> {result.commit_url}")


if __name__ == "__main__":
    main()
