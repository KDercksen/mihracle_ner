#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from json import dumps
from pathlib import Path
from typing import List

import numpy as np
import torch as t
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from utils import (
    biluo_to_transformer_examples,
    pad_sequence_to_length,
    read_gzip_json,
    write_gzip_json,
)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data_file", type=Path)
    p.add_argument("checkpoint_dir", type=Path)
    p.add_argument("--use-first-n-docs", type=int)
    p.add_argument("--outfile", type=Path)
    p.add_argument("--calculate-metrics", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    t.manual_seed(0)
    np.random.seed(0)

    dataset = read_gzip_json(args.data_file)
    if args.use_first_n_docs is not None:
        dataset = dataset[: args.use_first_n_docs]
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    model = AutoModelForTokenClassification.from_pretrained(
        args.checkpoint_dir.as_posix()
    )
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir.as_posix())
    _, test_data = biluo_to_transformer_examples(dataset, tokenizer)
    # input ids, input masks, segment ids and label ids respectively
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, collate_fn=lambda x: x
    )

    model.eval()
    results = []
    for batch in tqdm(test_loader):
        text, attention_masks, token_type_ids, _ = list(zip(*batch))
        max_len = max(len(t) for t in text)
        inputs = {}
        inputs["input_ids"] = pad_sequence_to_length(text, max_len, 0)
        inputs["attention_mask"] = pad_sequence_to_length(attention_masks, max_len, 0)
        inputs["token_type_ids"] = pad_sequence_to_length(token_type_ids, max_len, 0)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with t.no_grad():
            outputs = model(**inputs)
        preds = outputs[0].argmax(axis=2).detach().cpu().numpy()

        for i, sentence in enumerate(inputs["input_ids"]):
            pred_labels = [model.config.id2label[x] for x in preds[i, 1:-1]]
            words: List[str] = []
            labels: List[str] = []
            for word, label in zip(
                tokenizer.convert_ids_to_tokens(sentence, skip_special_tokens=True),
                pred_labels,
            ):
                # NOTE: this never happens for the first word, so empty list should work
                if word.startswith("##"):
                    words[-1] += word[2:]
                else:
                    words.append(word)
                    labels.append(label)
            results.append((words, labels))

    if args.outfile:
        write_gzip_json(args.outfile, results)
    else:
        print(dumps(results, indent=2))
