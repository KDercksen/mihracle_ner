#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from random import seed, shuffle

from seqeval.metrics.sequence_labeling import (
    f1_score,
    partial_f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from simstring_tool import (
    ConceptCharacterNgramFeatureExtractor,
    ConceptDictDatabase,
    ELModel,
)
from utils import read_json, write_json


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


# model params
save_path = Path("/mnt/storage/datasets/mihracle_data/el_model_snomednl.dill")
annotated_reports = Path(
    "/home/koen/projects/mihracle/data/preannotations/corrected_0-249.json"
)
lowercase = True
threshold = 0.97

p = argparse.ArgumentParser()
p.add_argument("--train", type=int, default=125)
p.add_argument("--nrepeats", type=int, default=10)
p.add_argument("--increments", type=int, default=5)
args = p.parse_args()

# 10x repetition to report confidence interval
use_for_train = args.train
n_repeats = args.nrepeats
increments = args.increments
seed(42)

reports = read_json(annotated_reports)

result_aggregator = []  # type: ignore
for i in range(n_repeats):
    print(f"Iteration {i+1}...")
    model = ELModel(save_path, lowercase=lowercase, topk=1, threshold=threshold)
    shuffle(reports)
    train = reports[:use_for_train]
    test = reports[use_for_train:]
    result = [test_model(model, test)]

    # update with increments of 5 reports
    for x in tqdm(range(0, len(train), increments), leave=False):
        model.update_db_with_annotations(train[x : x + increments], lowercase=lowercase)
        score = test_model(model, test)
        result.append(score)
    result_aggregator.append(result)
    # print(model.ignore_list)
    # print(model.add_list)
write_json(
    f"md_eval_{use_for_train}train_{n_repeats}repeats_{increments}increments.json",
    result_aggregator,
)
