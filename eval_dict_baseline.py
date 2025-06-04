#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
from itertools import chain
from pathlib import Path
from pprint import pprint

import numpy as np
from seqeval.metrics.sequence_labeling import classification_report, get_entities
from sklearn.model_selection import KFold

from utils import (
    build_entity_lookup_dict,
    concatenate_doc,
    find_subseq,
    overlaps,
    read_gzip_json,
    write_gzip_json,
)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("train_data", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("."))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--lowercase", action="store_true")
    args = p.parse_args()

    np.random.seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = read_gzip_json(args.train_data)

    folds = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    for i, (train_indices, val_indices) in enumerate(folds.split(data), start=1):
        print(f"{len(train_indices)} training examples...")
        print(f"{len(val_indices)} validation examples...")
        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]
        print("> Creating lookup dictionary...")
        lookup = build_entity_lookup_dict(train_data, lowercase=args.lowercase)

        val_data_concat = [concatenate_doc(doc) for doc in val_data]
        words, labels = list(zip(*val_data_concat))
        words = list(chain(*words))
        if args.lowercase:
            words = [w.lower() for w in words]
        labels = list(chain(*labels))
        entities_found = set()
        for key in lookup.keys():
            indices = find_subseq(words, list(key))
            if len(indices) > 0:
                for index in indices:
                    overlapping = [
                        x
                        for x in entities_found
                        if overlaps((index, index + len(key) - 1), x[1:])
                    ]
                    if len(overlapping) == 0 or all(
                        len(key) > (x[2] - x[1]) + 1 for x in overlapping
                    ):
                        type_name = random.choice(list(lookup[key]))
                        for x in overlapping:
                            entities_found.remove(x)
                        entities_found.add((type_name, index, index + len(key) - 1))

        actual_entities = set(get_entities(labels))
        evaluation = classification_report(
            actual_entities, entities_found, call_get_entities=False, output_json=True
        )
        pprint(evaluation)
        write_gzip_json(args.output_dir / f"val_scores_fold{i}.json.gz", evaluation)
