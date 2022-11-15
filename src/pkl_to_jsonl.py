#!/usr/bin/env python3

import argparse
import pickle
import json

args = argparse.ArgumentParser()
args.add_argument(
    "-i", "--input",
    default="data/cswiki-20221101-pages-articles-multistream.pkl"
)
args.add_argument(
    "-o", "--output",
    default="data/cswiki-20221101-pages-articles-multistream.jsonl"
)
args = args.parse_args()

with open(args.input, "rb") as f:
    data = pickle.load(f)

with open(args.output, "w") as f:
    for line in data.items():
        f.write(json.dumps(line, ensure_ascii=False)+ "\n")
