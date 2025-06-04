#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import warnings
from itertools import compress
from pathlib import Path
from typing import Dict, List, Tuple

import spacy
from spacy.training import offsets_to_biluo_tags

from utils import biluo_to_bio, clean_biluo, read_json, write_gzip_json


def load_spans(result):
    return [
        (x["value"]["start"], x["value"]["end"], x["value"]["labels"][0])
        for x in result
    ]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("train_data_dir", type=Path)
    p.add_argument("test_data", type=Path)
    p.add_argument("--span-only", action="store_true")
    p.add_argument("--bio", action="store_true")
    args = p.parse_args()

    name = "bio" if args.bio else "biluo"
    if args.span_only:
        name += "_span_only"
    print("Loading tokenizer and BRAT data...")
    nlp = spacy.load("nl_core_news_lg", disable=["ner", "tagger"])
    # load train
    train = [read_json(args.train_data_dir / f"{i}.json") for i in range(500)]
    # load test
    test = read_json(args.test_data)
    # truncate train (we annotate from the start of random sample)
    train = train[len(test) :]
    # [ doc0: [ ( sent0, { ... } ), ..., ( sentn, { ... } ) ], ... ]
    train_biluo: List[List[Tuple[List[str], Dict[str, List[str]]]]] = []
    test_biluo: List[List[Tuple[List[str], Dict[str, List[str]]]]] = []

    print("Converting annotations...")
    with warnings.catch_warnings():
        warnings.filterwarnings("once", category=UserWarning, module="spacy")
        for item in train:
            text = item["data"]["text"]  # type: ignore
            ents = load_spans(item["predictions"][0]["result"])
            parsed = nlp(text)
            tags = offsets_to_biluo_tags(parsed, ents)
            if args.bio:
                tags = biluo_to_bio(tags)
            else:
                tags = clean_biluo(tags)
            train_biluo.append([])
            for sent in parsed.sents:
                mask = [True] * len(sent)
                if tags[0][0] in ["I", "L"]:
                    # tie together "sentences" with boundary crossing entities
                    train_biluo[-1][-1][0].extend(
                        compress([token.text for token in sent], mask)
                    )
                    train_biluo[-1][-1][1]["entities"].extend(
                        compress(tags[: len(sent)], mask)
                    )
                else:
                    train_biluo[-1].append(
                        (
                            list(compress([token.text for token in sent], mask)),
                            {"entities": list(compress(tags[: len(sent)], mask))},
                        )
                    )
                # trim off tags belonging to the previous sentence
                tags = tags[len(sent) :]
        for item in test:
            text = item["data"]["text"]  # type: ignore
            ents = load_spans(item["annotations"][0]["result"])
            parsed = nlp(text)
            tags = offsets_to_biluo_tags(parsed, ents)
            if args.bio:
                tags = biluo_to_bio(tags)
            else:
                tags = clean_biluo(tags)
            test_biluo.append([])
            for sent in parsed.sents:
                mask = [True] * len(sent)
                if tags[0][0] in ["I", "L"]:
                    # tie together "sentences" with boundary crossing entities
                    test_biluo[-1][-1][0].extend(
                        compress([token.text for token in sent], mask)
                    )
                    test_biluo[-1][-1][1]["entities"].extend(
                        compress(tags[: len(sent)], mask)
                    )
                else:
                    test_biluo[-1].append(
                        (
                            list(compress([token.text for token in sent], mask)),
                            {"entities": list(compress(tags[: len(sent)], mask))},
                        )
                    )
                # trim off tags belonging to the previous sentence
                tags = tags[len(sent) :]

    print(f"Writing *.json.gz to {args.train_data_dir}/ ...")
    write_gzip_json(args.train_data_dir / f"train_{name}.json.gz", train_biluo)
    write_gzip_json(args.train_data_dir / f"test_{name}.json.gz", test_biluo)
