# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List, Dict
import torch
import logging
import numpy as np
import random
from transformers import PreTrainedTokenizer

from arguments import DataTrainingArguments
from datasets import load_dataset


def get_avg_results(results: List[dict]) -> dict:
    """
    Compute average results and standard deviation from many episodes.
    """
    aggregate_results = {'num_episodes': len(results)}

    for key in results[0]:
        try:
            numbers = np.array([res[key] for res in results])
            aggregate_results[key] = (numbers.mean(), numbers.std())

        except:
            pass

    return aggregate_results


def print_results(results: dict):
    for key, value in results.items():
        s = f'{key.replace("_", " "):26} '

        if isinstance(value, (list, tuple)):
            mean, std = value
            s += f'{mean:.6f} ± {std:.6f}'
        elif isinstance(value, float):
            s += f'{value:.6f}'
        else:
            s += f'{value}'

        logging.info(s)


def evaluate(model, dataset_name: str, data_args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, split: str,
             seed: int, gpu: int, batch_size: int, output_dir: str = None) -> Dict[str, float]:
    """
    Evaluate a model on some dataset.
    """
    model.eval()

    if gpu is None or gpu < 0 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", gpu)
    model.to(device)

    logging.info(f'Batch size: {batch_size}')
    logging.info(f'Num beams:  {data_args.num_beams}')
    logging.info(f'Max input length for evaluation:  {data_args.max_seq_length_eval}')
    logging.info(f'Max output length for evaluation: {data_args.max_output_seq_length_eval}')

    test_dataset = load_dataset(
        dataset_name, data_args,
        max_input_length=data_args.max_seq_length_eval,
        max_output_length=data_args.max_output_seq_length_eval,
        tokenizer=tokenizer, split=split, seed=seed, shuffle=False, is_eval=True,
    )

    if getattr(data_args, "eval_n_samples", 0) and data_args.eval_n_samples > 0 and len(test_dataset) > data_args.eval_n_samples:
        rng = random.Random(seed)
        sampled_positions = sorted(rng.sample(range(len(test_dataset.indices)), data_args.eval_n_samples))
        test_dataset.indices = [test_dataset.indices[i] for i in sampled_positions]
        test_dataset.effective_size = len(test_dataset.indices)
        logging.info(f"Eval subset enabled: sampled {test_dataset.effective_size} examples from split {split}")

    return test_dataset.evaluate_dataset(
        data_args=data_args,
        model=model,
        device=device,
        batch_size=batch_size,
        output_dir=output_dir,
    )
