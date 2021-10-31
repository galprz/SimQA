import argparse
import os
import random
import re


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OPTIONAL_VARS = ("VAR1", "VAR2", "VAR3", "VAR4", "VAR5", "VAR6", "VAR7")
MATH_OPERATORS = ("+", "-", "*", "/")


def assignment_no_others(x: str):
    return not any(a in x for a in ("repeat", "{", "}", "if", "(", ")", "return")) and "=" in x


def split_and_keep_numeric_str(x: str):
    return [c for c in x]


def assign_state(x: str):
    return "=" in x


def simple_assignment(x: str):
    return not any(op in x for op in MATH_OPERATORS)


def floor_div_assignment(x: str):
    return "/ /" in x


def aug_simple_assignment(state: str):
    var, value = state.split("=")

    var = var.strip()

    value = value.replace(" ", "")
    value = float(value)
    # if value % 1 == 0.0:
    #     value = int(value)

    half = value / 2
    half_str = split_and_keep_numeric_str(str(half))
    half_str = " ".join(half_str)

    double = 2 * value
    double_str = split_and_keep_numeric_str(str(double))
    double_str = " ".join(double_str)

    value_str = split_and_keep_numeric_str(str(value))
    value_str = " ".join(value_str)

    op = random.choice(MATH_OPERATORS)

    if op == "+":
        aug1 = f" {var} = {half_str} "
        aug2 = f" {var} = {var} + {half_str} "

    if op == "-":
        aug1 = f" {var} = {double_str} "
        aug2 = f" {var} = {var} - {value_str} "

    if op == "*":
        aug1 = f" {var} = {half_str} "
        aug2 = f" {var} = 2 * {var} "

    if op == "/":
        aug1 = f" {var} = {double_str} "
        aug2 = f" {var} = {var} / 2 "

    return aug1, aug2


def aug_floor_div_assignment(state: str):
    var, _ = state.split("=")

    op = random.choice(MATH_OPERATORS)

    if op == "+" or op == "-":
        aug1 = state
        aug2 = f" {var} = {var} {op} 0 "

    if op == "*" or op == "/":
        aug1 = state
        aug2 = f" {var} = {var} {op} 1 "

    return aug1, aug2


def aug_vars_assignment(state: str):
    var, x = state.split("=")

    for op in MATH_OPERATORS:
        try:
            ex1, ex2 = x.split(op)
        except Exception:
            pass

    aug1 = f" {var} = {ex1} + 0 "
    aug2 = state

    return aug1, aug2


def augment_tgt(tgt: str):

    states = tgt.split(";")
    x = [(s, i) for i, s in enumerate(states) if assignment_no_others(s)]
    state, ind = random.choice(x)

    if assign_state(state):
        if simple_assignment(state):
            aug1, aug2 = aug_simple_assignment(state)
        elif floor_div_assignment(state):
            aug1, aug2 = aug_floor_div_assignment(state)
        else:
            aug1, aug2 = aug_vars_assignment(state)
    else:
        aug1 = state
        aug2 = state

    states_tokens = [x.split() for x in states]
    aug1, aug2 = aug1.split(), aug2.split()

    new_states = states_tokens[:ind] + [aug1] + [aug2] + states_tokens[ind+1:]
    new_states = [x for x in new_states if len(x) > 0]
    new_states_str = [" ".join(x) for x in new_states]

    new_tgt = " ; ".join(new_states_str) + " ;"

    return new_tgt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="test", choices=("train", "test"))
    parser.add_argument("--version", type=str, default="v3", choices=("v1", "v2", "v3"))
    parser.add_argument("--shuffle", action="store_true", default=False)
    opts = parser.parse_args()

    processed_src_file = os.path.join(BASE_DIR, f"{opts.version}/processed/{opts.mode}/src.txt")
    with open(processed_src_file, "r") as f:
        processed_src = f.readlines()

    processed_tgt_file = os.path.join(BASE_DIR, f"{opts.version}/processed/{opts.mode}/tgt.txt")
    with open(processed_tgt_file, "r") as f:
        processed_tgt = f.readlines()

    new_srcs = []
    new_tgts = []
    for src, tgt in zip(processed_src, processed_tgt):

        # add original src and tgt
        # new_srcs.append(src)
        # new_tgts.append(tgt)

        aug_tgt = augment_tgt(tgt) + "\n"

        new_srcs.append(src)
        new_tgts.append(aug_tgt)

    x = list(zip(new_srcs, new_tgts))
    random.shuffle(x)
    new_srcs, new_tgts = zip(*x)

    with open(processed_src_file, "w") as f:
        f.writelines(new_srcs)

    with open(processed_tgt_file, "w") as f:
        f.writelines(new_tgts)
