#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch as t
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from utils import (
    EarlyStopper,
    SequenceScorer,
    biluo_to_transformer_examples,
    pad_sequence_to_length,
    read_gzip_json_files,
    write_gzip_json,
)


def filter_single_label(
    gold_truth: np.array,
    label_str: str,
    label_map: Dict[str, int],
    replace_value: int = -100,
):
    gold_truth_copy = gold_truth.copy()
    for key, val in label_map.items():
        if key == "O":
            continue
        if label_str not in key:
            gold_truth_copy[np.equal(gold_truth_copy, val)] = replace_value
    return gold_truth_copy


def save_checkpoint(
    args: argparse.Namespace,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    scorer,
):
    model.save_pretrained(args.checkpoint)
    tokenizer.save_pretrained(args.checkpoint)
    t.save(optimizer.state_dict(), args.checkpoint / "optimizer.pt")
    t.save(scheduler.state_dict(), args.checkpoint / "scheduler.pt")
    t.save(args, args.checkpoint / "training_args.bin")
    write_gzip_json(args.checkpoint / "scores.json.gz", scorer.scores)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-files", nargs="+", type=Path, required=True)
    p.add_argument("--use-first-n-docs", type=int)
    p.add_argument("--dev-files", nargs="+", type=Path, required=False)
    p.add_argument("--bio", action="store_true")
    p.add_argument("--freeze-base", action="store_true")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--num-warmup-steps", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epsilon", type=float, default=1e-8)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model-name", type=str, default="bert-base-multilingual-cased")
    args = p.parse_args()

    # set seeds
    np.random.seed(0)
    t.manual_seed(0)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    # make checkpoint dir
    args.checkpoint.mkdir(parents=True, exist_ok=True)

    name = "bio" if args.bio else "biluo"

    # processing training data
    train_data = read_gzip_json_files(*args.train_files)
    if args.use_first_n_docs is not None:
        train_data = train_data[: args.use_first_n_docs]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Converting training data to BERT format...")
    label_map, train_bert_data = biluo_to_transformer_examples(train_data, tokenizer)
    inv_label_map = {v: k for k, v in label_map.items()}

    # extract unique semantic types for per-label scores
    unique_semtypes = []
    for label_str in label_map.keys():
        if not label_str.startswith("O"):
            # skip the B-, I-, L-, U-
            if label_str[2:] not in unique_semtypes:
                unique_semtypes.append(label_str[2:])
    print(f"Unique semantic types extracted: {unique_semtypes}")

    train_dataloader = DataLoader(
        train_bert_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: x,
    )
    train_t_total = len(train_dataloader) * args.num_epochs

    modes = [(True, train_dataloader)]

    if args.dev_files is not None:
        print("Converting validation data to BERT format...")
        dev_data = read_gzip_json_files(*args.dev_files)
        _, dev_bert_data = biluo_to_transformer_examples(dev_data, tokenizer)

        dev_dataloader = DataLoader(
            dev_bert_data,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda x: x,
        )

        # add validation train mode + dataloader to the training loop
        modes.append((False, dev_dataloader))

    add_to_config = {
        "num_labels": len(label_map),
        "label2id": label_map,
        "id2label": inv_label_map,
    }
    config = AutoConfig.from_pretrained(args.model_name)
    config.update(add_to_config)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, config=config
    )
    model.to(device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    decaying_params = [
        p
        for n, p in model.named_parameters()
        if not any(nd in n for nd in no_decay)
        and (p not in model.base_model.parameters() if args.freeze_base else True)
    ]
    rest_params = [
        p
        for n, p in model.named_parameters()
        if any(nd in n for nd in no_decay)
        and (p not in model.base_model.parameters() if args.freeze_base else True)
    ]
    optimizer_grouped_parameters = [
        {"params": decaying_params, "weight_decay": args.weight_decay},
        {"params": rest_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=train_t_total,
    )
    earlystopper = EarlyStopper(mode="max", checkpoint_callback=save_checkpoint)
    for it in range(args.num_epochs):
        print(f"Epoch #{it+1}...")
        for mode, loader in modes:
            model.train(mode=mode)
            t.set_grad_enabled(mode)
            scorer = SequenceScorer()
            for batch in tqdm(loader, leave=False):
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
                inputs["labels"] = pad_sequence_to_length(labels, max_len, -100)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                loss = outputs[0]

                if mode:
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                predictions = outputs[1].argmax(axis=2).detach().cpu().numpy()
                gold_labels = inputs["labels"].detach().cpu().numpy()
                scorer.update(predictions, gold_labels, inv_label_map)

            print("------------------------------")
            print("{} scores:".format("Training" if mode else "Validation"))
            print(scorer)
            print("++++++++++++++++++++++++++++++")
        # early stopping based on the F-score, save checkpoint etc when improved over
        # best known
        if earlystopper(
            scorer.scores["micro_avg"]["f1"],
            args,
            model,
            tokenizer,
            optimizer,
            scheduler,
            scorer,
        ):
            print(f"F1 has not improved from {earlystopper.best_value:.4f}, stopping!")
