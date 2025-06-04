#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from random import seed, shuffle

import spacy
import torch as t
from flair.data import Corpus, FlairDataset, Sentence, Token
from flair.embeddings import (
    CharacterEmbeddings,
    FlairEmbeddings,
    StackedEmbeddings,
    TransformerWordEmbeddings,
    WordEmbeddings,
)
from flair.models import SequenceTagger
from flair.models.sequence_tagger_utils.bioes import get_spans_from_bio
from flair.trainers import ModelTrainer
from seqeval.metrics.sequence_labeling import (
    f1_score,
    partial_f1_score,
    precision_score,
    recall_score,
)
from spacy.training import offsets_to_biluo_tags
from torch.optim.adam import Adam
from tqdm import tqdm

from convert_labelstudio_reports_to_biluo import load_spans
from simstring_tool import (
    ConceptCharacterNgramFeatureExtractor,
    ConceptDictDatabase,
    ELModel,
)
from utils import biluo_to_bio, read_json, read_jsonl, write_json


def test_model(model, test_reports):
    predictions = model.predict(test_reports)
    predicted_entities = []
    real_entities = []
    for i, (prediction, report) in enumerate(zip(predictions[1], test_reports)):
        predicted_entities.extend(
            (i, x["value"]["labels"][0], x["value"]["start"], x["value"]["end"])
            for x in prediction["result"]
        )
        real_entities.extend(
            (i, x["value"]["labels"][0], x["value"]["start"], x["value"]["end"])
            for x in report["annotations"][0]["result"]
        )
    return {
        "f1_score": f1_score(
            set(real_entities), set(predicted_entities), call_get_entities=False
        ),
        "precision_score": precision_score(
            set(real_entities), set(predicted_entities), call_get_entities=False
        ),
        "recall_score": recall_score(
            set(real_entities), set(predicted_entities), call_get_entities=False
        ),
        # NOTE: comment this out for now, it's very slow and incorrectly implemented
        # "partial_f1_score": partial_f1_score(
        #     set(real_entities), set(predicted_entities), call_get_entities=False
        # ),
    }


def ls_to_flair(report, key="annotations"):
    text = nlp(report["data"]["text"])
    spans = load_spans(report[key][0]["result"])
    biluo_tags = offsets_to_biluo_tags(text, spans)
    bio_tags = biluo_to_bio(biluo_tags)

    filtered_tokens, filtered_tags = zip(
        *[(t.text, tag) for t, tag in zip(text, bio_tags) if not t.text.isspace()]
    )

    s = Sentence(list(filtered_tokens))
    for span, _, label in get_spans_from_bio(list(filtered_tags)):
        s[span[0] : span[-1] + 1].add_label("ner", label)
    return s


class SimpleDataset(FlairDataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __getitem__(self, i):
        return self.sentences[i]

    def __len__(self):
        return len(self.sentences)


# model params
save_path = Path("/mnt/storage/datasets/mihracle_data/el_model_snomednl.dill")
annotated_reports = Path(
    "/home/koen/projects/mihracle/data/preannotations/corrected_0-249.json"
)
lowercase = True
threshold = 0.97

# Arguments
p = argparse.ArgumentParser()
p.add_argument("output_dir", type=Path)
p.add_argument("--train", type=int, default=125)
p.add_argument("--nrepeats", type=int, default=10)
p.add_argument("--increments", type=int, default=5)

args = p.parse_args()

args.output_dir.mkdir(parents=True, exist_ok=True)

use_for_train = args.train
n_repeats = args.nrepeats
increments = args.increments
seed(42)
t.manual_seed(42)
t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False

print("Loading reports...")
reports = read_json(annotated_reports)
extra_reports_250 = [{"data": x} for x in read_jsonl("data/500_reports.jsonl")[250:]]
extra_reports_1000 = [{"data": x} for x in read_jsonl("data/1000_random_reports.jsonl")]

print("Loading spacy model...")
nlp = spacy.load("nl_core_news_lg", disable=["ner", "tagger"])

result_aggregator = []  # type: ignore

for i in range(n_repeats):
    print(f"Iteration {i+1}...")
    print("Loading ELModel...")
    model = ELModel(save_path, lowercase=lowercase, topk=1, threshold=threshold)
    shuffle(reports)
    train = reports[:use_for_train]
    test = [ls_to_flair(x, key="annotations") for x in reports[use_for_train:]]
    # result = [test_model(model, test)]

    for extraname, extra in zip(
        # ["none", "250", "1000"], [[], extra_reports_250, extra_reports_1000]
        ["250", "1000"],
        [extra_reports_250, extra_reports_1000],
    ):
        # update with increments of inc reports
        for x in tqdm(range(0, len(train), increments), leave=False):
            print(f"Update {x} - {x+increments}...")
            model.update_db_with_annotations(
                train[x : x + increments], lowercase=lowercase
            )
            # predict train reports with updated model
            print("predicting...")
            pred_reports, predictions = model.predict(train + extra)  # type: ignore
            train_sents = [
                ls_to_flair({"data": r, "predictions": [p]}, key="predictions")
                for r, p in zip(pred_reports, predictions)
            ]
            corpus = Corpus(
                SimpleDataset(train_sents),
                dev=SimpleDataset(test),
                test=SimpleDataset(test),
            )
            embeddings = StackedEmbeddings(
                embeddings=[
                    # WordEmbeddings(
                    #     "/mnt/storage/datasets/mihracle_data/models/"
                    #     "w2v_dumps_thorax_full/fasttext_cbow_300_5epochs.bin"
                    # ),
                    # CharacterEmbeddings(),
                    # FlairEmbeddings("nl-forward"),
                    # FlairEmbeddings("nl-backward"),
                    TransformerWordEmbeddings(
                        "bert-base-multilingual-cased", fine_tune=True
                    ),
                ]
            )
            tagger = SequenceTagger(
                hidden_size=256,
                embeddings=embeddings,
                tag_dictionary=corpus.make_label_dictionary("ner"),
                tag_type="ner",
                use_crf=True,
            )
            trainer = ModelTrainer(
                tagger,
                corpus,
            )
            print("Training flair model...")
            trainer.train(
                args.output_dir
                / f"test_flair_ner_biobert_repition{i}_update{x}_extra{extraname}",
                learning_rate=0.1,
                mini_batch_size=2,
                mini_batch_chunk_size=1,  # just to make sure we do not error out...
                max_epochs=90,
                use_amp=True,
            )
        # score = test_model(model, test)
        # result.append(score)
    # result_aggregator.append(result)
    # print(model.ignore_list)
    # print(model.add_list)
# write_json(
#     f"md_eval_{use_for_train}train_{n_repeats}repeats_{increments}increments.json",
#     result_aggregator,
# )
