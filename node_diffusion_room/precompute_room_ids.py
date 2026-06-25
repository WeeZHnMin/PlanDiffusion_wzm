"""
预计算所有样本的 room_ids 并写入新 npz。

用法：
  python -m node_diffusion_rid.precompute_room_ids \\
      --inp data/processed/node_diffusion_cross_att/graph_dataset.npz \\
      --out data/processed/node_diffusion_rid/graph_dataset.npz

输出 npz 包含原有全部 key，再加：
  room_ids  [N, 40]  int32，值域 [0, MAX_ROOMS]
             0 = 不属于任何环（孤立 / 填充节点）
"""

import argparse
import os
from multiprocessing import Pool, cpu_count

import numpy as np

from .model import _assign_room_ids_single, MAX_ROOMS


def _worker(args):
    idx, adj_row, mask_row = args
    n = int(mask_row.sum())
    if n < 3:
        return idx, [0] * adj_row.shape[0]
    ids = _assign_room_ids_single(adj_row[:n, :n].astype(bool), n)
    # 补齐到 max_nodes 长度（填充位保持 0）
    full = [0] * adj_row.shape[0]
    full[:n] = ids
    return idx, full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp', default='data/processed/node_diffusion_cross_att/graph_dataset.npz')
    parser.add_argument('--out', default='data/processed/node_diffusion_rid/graph_dataset.npz')
    parser.add_argument('--workers', type=int, default=0, help='0=cpu_count()-1')
    args = parser.parse_args()

    print(f'读取: {args.inp}')
    d = np.load(args.inp, allow_pickle=True)
    adj_matrix = d['adj_matrix']   # [N, 40, 40]
    node_mask  = d['node_mask']    # [N, 40]
    N, max_nodes, _ = adj_matrix.shape

    n_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 1)
    print(f'共 {N} 条样本，workers={n_workers}')

    tasks = [(i, adj_matrix[i], node_mask[i]) for i in range(N)]

    room_ids_out = np.zeros((N, max_nodes), dtype=np.int32)
    with Pool(processes=n_workers) as pool:
        for i, (idx, ids) in enumerate(pool.imap_unordered(_worker, tasks, chunksize=256)):
            room_ids_out[idx] = ids
            if (i + 1) % 10000 == 0:
                print(f'  {i+1}/{N}', flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # 把原有所有 key 加上 room_ids 一起写进新 npz
    save_dict = {k: d[k] for k in d.files}
    save_dict['room_ids'] = room_ids_out
    np.savez(args.out, **save_dict)
    print(f'完成 → {args.out}')
    print(f'room_ids shape: {room_ids_out.shape}  max_id: {room_ids_out.max()}')


if __name__ == '__main__':
    main()
