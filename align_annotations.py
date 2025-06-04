#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import numpy as np
import spacy

from utils import read_jsonl, write_jsonl

nlp = spacy.load("nl_core_news_lg", disable=["ner", "tagger", "parser"])


def align(clean, dirty, labels):
    assert len(clean) <= len(dirty)
    alignment = np.zeros(len(dirty))
    offset = 0
    # build offset list
    for i, c_clean in enumerate(clean):
        if dirty[i + offset] == c_clean:
            alignment[i + offset] = offset
        else:
            offset += 1

    # update labels accordingly
    new_labels = []
    for start, end, label in labels:
        new_start = int(start - alignment[start])
        new_end = int(end - alignment[start])
        new_labels.append((new_start, new_end, label))

    return {"text": clean, "labels": new_labels}


def clean_text(text):
    # taken from diag-radiology-report-anonymizer
    text = re.sub(" +", " ", text)
    text = re.sub(" *\n *", "\n", text)
    newlines = re.search("\n+", text)
    if newlines is not None:
        if newlines.start() == 0:
            text = re.sub(newlines.group(), "", text, 1)
    text = re.sub(" *\n *", "\n", text)
    text = re.sub(" *- *", "-", text)
    text = re.sub(" */ *", "/", text)
    return text


def clean_text_for_simstring_comparison(text):
    # replace newlines with spaces
    return re.sub("\n+", "\n", text)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("rra_data", type=Path)
    p.add_argument("rra_data_out", type=Path)
    args = p.parse_args()

    rra_data = read_jsonl(args.rra_data)
    new_rra_data = [
        align(clean_text_for_simstring_comparison(x["text"]), x["text"], x["labels"])
        for x in rra_data
    ]

    write_jsonl(args.rra_data_out, new_rra_data)
