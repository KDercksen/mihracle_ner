#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import spacy
import torch as t
from flair.data import Corpus, FlairDataset, Sentence, Token
from flair.embeddings import (
    CharacterEmbeddings,
    FlairEmbeddings,
    StackedEmbeddings,
    WordEmbeddings,
)
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from sklearn.model_selection import KFold

from utils import biluo_to_bio, read_gzip_json


def to_flair_sentence(example):
    # example:
    #    List[
    #        Tuple[
    #            List[str] < tokens, e.g. ["Hi", "I", "am", "Koen"]
    #            Dict[str, List[str]] < label dict, e.g. {"entities": ["O", "O", "O", "PERSON"]}
    #        ]
    #    ]
    # returns Sentence objects, Sentence is essentially a list of tokens
    # each token has a tag of type "ner" with the value "O" or "B-PERSON" or whatever
    # see also
    # https://github.com/flairNLP/flair/blob/master/flair/data.py#L264 (token)
    # https://github.com/flairNLP/flair/blob/master/flair/data.py#L529 (sentence)
    s = Sentence()
    for sent in example:
        tokens, tags = sent
        tags = biluo_to_bio(tags["entities"])
        for token, tag in zip(tokens, tags):
            if not token.isspace():
                t = Token(token)
                t.add_tag("ner", tag)
                s.add_token(t)
    return s


class SimpleDataset(FlairDataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __getitem__(self, idx):
        return self.sentences[idx]

    def __len__(self):
        return len(self.sentences)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("train_data", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("."))
    p.add_argument("--iter", type=int, default=1000)
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()

    np.random.seed(0)
    t.manual_seed(0)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False

    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = read_gzip_json(args.train_data)

    data = [to_flair_sentence(report) for report in data]
    tag_dict = Corpus(data).make_tag_dictionary("ner")

    folds = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    for fold_idx, (train_indices, val_indices) in enumerate(folds.split(data), start=1):
        print(f"{len(train_indices)} training examples...")
        print(f"{len(val_indices)} validation examples...")
        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]

        # create a corpus for training a SequenceTagger model
        corpus = Corpus(SimpleDataset(train_data), test=SimpleDataset(val_data))
        embeddings = StackedEmbeddings(
            embeddings=[
                # stack some embeddings; see Flair tutorials
                WordEmbeddings("nl"),
                CharacterEmbeddings(),
                FlairEmbeddings("nl-forward"),
                FlairEmbeddings("nl-backward"),
            ]
        )
        # build SequenceTagger, see also Flair tutorials/source for parameters etc
        tagger = SequenceTagger(
            hidden_size=256,
            embeddings=embeddings,
            tag_dictionary=tag_dict,
            tag_type="ner",
            use_crf=True,
        )
        # use the trainer
        trainer = ModelTrainer(tagger, corpus)
        trainer.train(
            args.output_dir / f"ct-thorax-ner-fold{fold_idx}",
            learning_rate=0.1,
            mini_batch_size=32,
            max_epochs=args.iter,
        )
