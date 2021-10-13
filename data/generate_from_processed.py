import argparse
import json
import os
import re
import torch

from onmt.inputters.dataset_base import Dataset
from onmt.inputters.datareader_base import DataReaderBase
from torchtext.data.example import Example


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Reader(DataReaderBase):

    def read(self, data, side, src_dir=None):
        new_data = {}
        new_data["src"] = data["context"]
        new_data["tgt"] = data["code"]
        new_data["indices"] = int(side)
        yield new_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="test", choices=("train", "test"))
    parser.add_argument("--version", type=str, default="v3", choices=("v1", "v2", "v3"))
    opts = parser.parse_args()

    train_path = os.path.join(BASE_DIR, f"{opts.version}/{opts.mode}.json")
    with open(train_path, "r") as f:
        file_data = json.load(f)

    ds_version = file_data["version"]
    data = file_data["data"]
    nb_samples = len(data)

    reader = Reader()
    ds = Dataset(
        fields={},
        readers=nb_samples * [reader],
        data=[(str(i), x) for i, x in enumerate(data)],
        dirs=[i for i in range(nb_samples)],
        sort_key=None,
        corpus_id=opts.mode,
    )

    processed_src_file = os.path.join(BASE_DIR, f"{opts.version}/processed/{opts.mode}/src.txt")
    with open(processed_src_file, "r") as f:
        processed_src = f.readlines()

    processed_tgt_file = os.path.join(BASE_DIR, f"{opts.version}/processed/{opts.mode}/tgt.txt")
    with open(processed_tgt_file, "r") as f:
        processed_tgt = f.readlines()


    ############### HACK
    ds.examples = []
    for i, src_tgt in enumerate(zip(processed_src, processed_tgt)):
        src, tgt = src_tgt
        ex = Example()
        ex.__setattr__("src", [src.split()])
        ex.__setattr__("tgt", [tgt.split()])
        ex.__setattr__("indices", i)
        ex.__setattr__("corpus_id", opts.mode)
        ds.examples.append(ex)

    ds.__setattr__("fields", [])

    save_as = "valid" if opts.mode is "test" else opts.mode
    ds.save(f"{opts.version}/processed/SimQA.{save_as}.0.pt")
