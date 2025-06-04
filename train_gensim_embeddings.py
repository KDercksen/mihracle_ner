#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

from gensim.models import FastText, Word2Vec
from gensim.utils import simple_preprocess

MODELS = {
    "word2vec": Word2Vec,
    "fasttext": FastText,
}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path)
    p.add_argument("--size", type=int, default=100)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--skip-gram", action="store_true")
    p.add_argument("--cbow", action="store_true")
    p.add_argument("--model", default="word2vec", choices=MODELS.keys())
    p.add_argument("--save", type=Path, required=True)
    p.add_argument("--w2v-format", action="store_true")
    args = p.parse_args()

    # cbow takes precedence, skip-gram is default; e.g.:
    # --cbow --skip-gram == --cbow
    # --cbow == --cbow
    # --skip-gram == --skip-gram
    # <nothing> == --skip-gram
    args.skip_gram = not args.cbow

    with open(args.path) as f:
        line_iterator = [simple_preprocess(line, max_len=30) for line in f]
    model_class = MODELS[args.model]

    model = model_class(
        sentences=line_iterator,
        vector_size=args.size,
        window=5,
        min_count=1,
        workers=10,
        epochs=args.epochs,
        sg=args.skip_gram,
    )
    if args.w2v_format:
        model.wv.save_word2vec_format(args.save.as_posix())
    else:
        model.save(args.save.as_posix())
