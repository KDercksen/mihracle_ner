#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import torch as t
from sklearn.model_selection import KFold
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from utils import (
    SequenceScorer,
    TransformerNERDataset,
    biluo_to_transformer_examples,
    pad_sequence_to_length,
    read_gzip_json_files,
    write_gzip_json,
)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-files", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("."))
    p.add_argument("--tokenizer", type=str, default="bert-base-multilingual-cased")
    args = p.parse_args()

    np.random.seed(0)
    t.manual_seed(0)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False

    if not args.output_dir.exists():
        raise ValueError("Incorrect path supplied")

    num_folds = len(list(args.output_dir.glob("model_checkpoint_*")))
    assert num_folds > 0, "Cannot evaluate without model checkpoints :("

    # set seeds
    np.random.seed(0)
    t.manual_seed(0)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    train_data = read_gzip_json_files(*args.train_files)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print("Converting training data to BERT format...")
    _, bert_data = biluo_to_transformer_examples(train_data, tokenizer)

    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    for i, (train_indices, val_indices) in enumerate(kfold.split(bert_data), start=1):
        print(val_indices)
        dev_bert_data = [bert_data[i] for i in val_indices]
        dev_dataset = TransformerNERDataset(dev_bert_data)
        model = AutoModelForTokenClassification.from_pretrained(
            str(args.output_dir / f"model_checkpoint_{i}")
        )
        model.to(device)

        dev_dataloader = DataLoader(
            dev_dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=lambda x: x,
        )

        def write_output(scorer, data):
            # TODO: change this checkpoint
            write_gzip_json(
                args.output_dir / f"val_only_scores_fold{i}.json.gz", scorer.scores
            )
            write_gzip_json(
                args.output_dir / f"model_checkpoint_{i}_preds.json.gz", data
            )

        model.eval()
        scorer = SequenceScorer()
        token_collection = []
        for batch in tqdm(dev_dataloader, leave=False):
            text, attention_masks, token_type_ids, labels = list(zip(*batch))
            max_len = max(len(t) for t in text)
            inputs = {}
            inputs["input_ids"] = pad_sequence_to_length(text, max_len, 0)
            inputs["attention_mask"] = pad_sequence_to_length(
                attention_masks, max_len, 0
            )
            inputs["token_type_ids"] = pad_sequence_to_length(
                token_type_ids, max_len, 0
            )
            inputs["labels"] = pad_sequence_to_length(labels, max_len)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with autocast():
                outputs = model(**inputs)

            predictions = outputs[1].argmax(axis=2).detach().cpu().numpy()
            gold_labels = inputs["labels"].detach().cpu().numpy()
            scorer.update(predictions, gold_labels, model.config.id2label)
            token_collection.extend([tokenizer.convert_ids_to_tokens(t) for t in text])

        fixed_token_collection = []
        for token_c in token_collection:
            words = []
            for word in token_c[1:-1]:
                if word.startswith("##"):
                    words[-1] += word[2:]
                else:
                    words.append(word)
            fixed_token_collection.append(words)
        write_output(
            scorer,
            list(zip(fixed_token_collection, scorer.gold_labels, scorer.predictions)),
        )
