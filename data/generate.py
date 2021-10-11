import argparse
import json
import os
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
        new_data["tgt"] =  data["code"]
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

    train_data_file = os.path.join(BASE_DIR, f"{opts.version}/processed/SimQA.valid.0.pt")
    example = torch.load(train_data_file)

    context_tokenizer = get_tokenizer(tokenizer="toktok", language='en')
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
    print(ds)
    # ds.save("TODO")

    print()
