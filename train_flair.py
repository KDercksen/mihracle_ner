#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flair.datasets import ColumnCorpus
from flair.embeddings import (
    FlairEmbeddings,
    StackedEmbeddings,
    TransformerWordEmbeddings,
)
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

columns = {0: "text", 3: "ner"}
corpus = ColumnCorpus("./", columns, train_file="./data/simplerad_annotated.conll")

label_dict = corpus.make_label_dictionary(label_type="ner")

embedding_types = [
    TransformerWordEmbeddings(
        "./models/bert-base-multilingual-cased-finetuned-full",
        layers="-1",
        layer_mean=False,
        fine_tune=False,
    ),
    # FlairEmbeddings("nl-forward"),
    # FlairEmbeddings("nl-backward"),
]
embeddings = StackedEmbeddings(embeddings=embedding_types)

tagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=label_dict,
    tag_type="ner",
    use_crf=True,
)

trainer = ModelTrainer(tagger, corpus)

trainer.train(
    "models/flair_simplerad", learning_rate=0.1, mini_batch_size=2, max_epochs=150
)
