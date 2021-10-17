import argparse
import os
import random


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OPTIONAL_VARS = ("VAR1", "VAR2", "VAR3", "VAR4", "VAR5")
MATH_OPERATORS = ("+", "-", "*", "/", "//")


def split_and_keep_numeric_str(x: str):
    return [c for c in x]


def rand_numeric():
    base_int = random.randint(0, 100)
    residual = random.randint(0, 100) / 100.
    return base_int + residual


def rand_math_operator():
    return random.choice(MATH_OPERATORS)


def random_single_aug(curr_vars: set):
    if len(curr_vars) == 0:
        var = random.choice(OPTIONAL_VARS)
        curr_vars.add(var)

        num = rand_numeric()
        num_as_str_list = [c for c in str(num)]

        return [var, "="] + num_as_str_list + [";"]

    else:
        vars_ = list(curr_vars)

        var1 = random.choice(OPTIONAL_VARS)
        curr_vars.add(var1)

        if random.random() < 0.5:
            x = [random.choice(vars_)]
        else:
            num = rand_numeric()
            x = [c for c in str(num)]

        y = [rand_math_operator()]

        if random.random() < 0.5:
            z = [random.choice(vars_)]
        else:
            num = rand_numeric()
            z = [c for c in str(num)]

        return [var1, "="] + x + y + z + [";"]


def random_augs(max_nb_augs: int = 2):

    vars_set = set()

    nb_augs = random.randint(0, max_nb_augs)
    augs_list = []
    for _ in range(nb_augs):
        aug = random_single_aug(vars_set)
        augs_list.extend(aug)
    augs = " ".join(augs_list)
    return augs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="test", choices=("train", "test"))
    parser.add_argument("--version", type=str, default="v3", choices=("v1", "v2", "v3"))
    parser.add_argument("--max-nb-augs", type=int, default=2)
    opts = parser.parse_args()

    processed_tgt_file = os.path.join(BASE_DIR, f"{opts.version}/processed/{opts.mode}/tgt.txt")
    with open(processed_tgt_file, "r") as f:
        processed_tgt = f.readlines()

    new_tgts = []
    for tgt in processed_tgt:
        augs = random_augs(opts.max_nb_augs)
        if augs is not "":
            new_tgts.append(augs + " " + tgt)
        else:
            new_tgts.append(tgt)

    with open(processed_tgt_file, "w") as f:
        f.writelines(new_tgts)
