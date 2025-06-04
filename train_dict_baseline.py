#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

from utils import build_entity_lookup_dict, read_gzip_json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("train_data", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("."))
    p.add_argument("--use-first-n-docs", type=int)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = read_gzip_json(args.train_data)
    if args.use_first_n_docs is not None:
        data = data[: args.use_first_n_docs]

    print("> Creating lookup dictionary...")
    lookup = build_entity_lookup_dict(data)

    print(f"> Storing lookup dictionary in {args.output_dir}/lookup_dict.json...")
    with open(args.output_dir / "lookup_dict.json", "w") as f:
        json.dump([(list(k), list(v)) for k, v in lookup.items()], f, indent=2)
