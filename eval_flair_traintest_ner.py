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


def new_model(embeddings, tag_dict):
    return SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dict,
        tag_type="ner",
        use_crf=True,
    )


def pretrained_model_with_new_head(tag_dict):
    # see also this issue for explanation:
    # https://github.com/flairNLP/flair/issues/1797
    # NOTE: i havent tested this myself
    tagger = SequenceTagger.load("flair/ner-english")
    # see: https://github.com/flairNLP/flair/blob/master/flair/models/sequence_tagger_model.py#L26
    new_tagger = SequenceTagger(
        hidden_size=256,
        embeddings=tagger.embeddings,
        tag_dictionary=tag_dict,
        tag_type="ner",
    )
    # reuse old internal layers
    new_tagger.embedding2nn = tagger.embedding2nn
    new_tagger.rnn = tagger.rnn
    return new_tagger


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("train_data", type=Path)
    p.add_argument("test_data", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("."))
    p.add_argument("--iter", type=int, default=1000)
    args = p.parse_args()

    np.random.seed(0)
    t.manual_seed(0)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_data = read_gzip_json(args.train_data)
    test_data = read_gzip_json(args.test_data)

    train_data = [to_flair_sentence(report) for report in train_data]
    test_data = [to_flair_sentence(report) for report in test_data]
    tag_dict = Corpus(train_data).make_tag_dictionary("ner")

    # create a corpus for training a SequenceTagger model
    corpus = Corpus(SimpleDataset(train_data), test=SimpleDataset(test_data))
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
    # NOTE: this is from scratch!
    tagger = new_model(embeddings, tag_dict)
    # tagger = pretrained_model_with_new_head(tag_dict)
    # use the trainer
    trainer = ModelTrainer(tagger, corpus)
    trainer.train(
        args.output_dir,
        learning_rate=0.1,
        mini_batch_size=32,
        max_epochs=args.iter,
    )
