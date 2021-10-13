import argparse
import json
import os
import re
import torch

from grammer.tokenizer import SimCodeTokenizer
from onmt.inputters.dataset_base import Dataset
from onmt.inputters.datareader_base import DataReaderBase
from torchtext.data.utils import get_tokenizer
from torchtext.data.example import Example
from torchtext.data.field import Field


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Reader(DataReaderBase):

    def read(self, data, side, src_dir=None):
        new_data = {}
        new_data["src"] = data["context"]
        new_data["tgt"] = data["code"]
        new_data["indices"] = int(side)
        yield new_data


class TokenizeField_(Field):

    def set_tokenizer(self, tokenize_fn):
        self.__setattr__("tokenize_fn", tokenize_fn)

    def preprocess(self, x):
        return [self.tokenize_fn(x)]


class PassField_(Field):

    def preprocess(self, x):
        return x


class Tokenizer:

    def apply_and_flatten(self, fn, x):
        _parts_ = []
        [_parts_.extend(fn(i)) if isinstance(fn(i), list) else _parts_.append(i) for i in x]
        clean_parts = [i for i in _parts_ if i != ""]
        return clean_parts

    def split_and_keep_numeric(self, x: str):
        return [c for c in x] if re.match(r'^-?\d+(?:\.\d+)$', x) or x.isdigit() else x

    def split_and_keep_suffix(self, x: str):
        return x if x[-1] not in ("%", "°", ".", "!", "?") else [x[:-1], x[-1]]

    def split_and_keep_parentheses(self, x: str):
        if len(x) == 1:
            return x

        x_ = x
        if x[0] in ("(", "[", "{"):
            x_ = [x[0], x[1:]]
        if x[-1] in (")", "]", "}"):
            x_ = [x_[0], x_[1][:-1], x_[1][-1]]
        return x_

    def split_and_keep_dash(self, x: str):
        return x if "-" not in x else re.split("(-)", x)

    def __call__(self, x: str):
        tokens = x.split()
        tokens = self.apply_and_flatten(self.split_and_keep_parentheses, tokens)
        tokens = self.apply_and_flatten(self.split_and_keep_dash, tokens)
        tokens = self.apply_and_flatten(self.split_and_keep_numeric, tokens)
        tokens = self.apply_and_flatten(self.split_and_keep_suffix, tokens)

        return tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=("train", "test"))
    parser.add_argument("--version", type=str, default="v3", choices=("v1", "v2", "v3"))
    opts = parser.parse_args()

    train_path = os.path.join(BASE_DIR, f"{opts.version}/{opts.mode}.json")
    with open(train_path, "r") as f:
        file_data = json.load(f)

    ds_version = file_data["version"]
    data = file_data["data"]
    nb_samples = len(data)

    train_data_file = os.path.join(BASE_DIR, f"{opts.version}/processed/SimQA.valid.0.pt")
    example = torch.load(train_data_file)

    # context_tokenizer = get_tokenizer(tokenizer="toktok", language='en')
    context_tokenizer = Tokenizer()
    sim_qa_tokenizer = SimCodeTokenizer()

    reader = Reader()

    context_field = TokenizeField_()
    context_field.set_tokenizer(context_tokenizer)

    sim_qa_field = TokenizeField_()
    sim_qa_field.set_tokenizer(sim_qa_tokenizer.tokenize)

    ds = Dataset(
        fields={
            # "src": [("src", context_field)],
            # "tgt": [("tgt", sim_qa_field)],
            # "indices": [("indices", Field())],
            # "corpus_id": [("corpus_id", Field())],
        },
        readers=nb_samples * [reader],
        data=[(str(i), x) for i, x in enumerate(data)],
        dirs=[i for i in range(nb_samples)],
        sort_key=None,
        corpus_id=opts.mode,
    )

    ############### HACK
    ds.examples = []
    for i, x in enumerate(data):
        ex = Example()
        ex.__setattr__("src", [context_tokenizer(x["context"])])
        ex.__setattr__("tgt", [sim_qa_tokenizer.tokenize(x["code"])])
        ex.__setattr__("indices", i)
        ex.__setattr__("corpus_id", opts.mode)
        ds.examples.append(ex)

    ds.__setattr__("fields", [])
    # ds.save(f"{opts.version}/processed/SimQA.{opts.mode}.0.pt")
