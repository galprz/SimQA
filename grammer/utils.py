import sys
from collections import Counter
from enum import Enum
from typing import List, Tuple
import re
import math

from antlr4 import InputStream, ParseTreeWalker
from io import StringIO
from antlr4.error.ErrorListener import ConsoleErrorListener
from grammer.src.Sim2GenericListener import Sim2GenericListener
from grammer.src.Sim2PythonListener import Sim2PythonListener
from grammer.src.SimCodeLexer import SimCodeLexer, CommonTokenStream
from grammer.src.SimCodeParser import SimCodeParser
from grammer.src.SimPrinterListener import SimPrinterListener
from grammer.src.SimTokensListener import Sim2TokensListener
from torchtext.vocab import Vocab


class BlockType(Enum):
    NotInBlock = (1,)
    IfBlock = (2,)
    WhileBlock = (3,)
    Assignment = 4
    Return = 5


def sim2python(sim_code: str, add_traces=False, silence=True):
    sim_code = wrap_body(sim_code)
    chars = InputStream(sim_code)
    lexer = SimCodeLexer(chars)
    if silence:
        lexer.removeErrorListener(ConsoleErrorListener.INSTANCE)
    tokens = CommonTokenStream(lexer)
    parser = SimCodeParser(tokens)
    if silence:
        parser.removeErrorListener(ConsoleErrorListener.INSTANCE)
    tree = parser.parse()
    sim2python_listener = Sim2PythonListener(add_traces)
    walker = ParseTreeWalker()
    walker.walk(sim2python_listener, tree)
    return sim2python_listener.get_generated_code()


def normalize_sim_code(sim_code: str):
    sim_code = wrap_body(sim_code)
    chars = InputStream(sim_code)
    lexar = SimCodeLexer(chars)
    tokens = CommonTokenStream(lexar)
    parser = SimCodeParser(tokens)
    tree = parser.parse()
    sim2generic_listener = Sim2GenericListener()
    walker = ParseTreeWalker()
    walker.walk(sim2generic_listener, tree)
    return sim2generic_listener.get_generated_code()


def tokenizer(sim_code: str, stoi) -> List[str]:
    sim_code = wrap_body(sim_code)
    chars = InputStream(sim_code)
    lexar = SimCodeLexer(chars)
    tokens = CommonTokenStream(lexar)
    parser = SimCodeParser(tokens)
    tree = parser.parse()
    sim_toknizer = Sim2TokensListener()
    walker = ParseTreeWalker()
    walker.walk(sim_toknizer, tree)
    tokens = sim_toknizer.get_tokens()
    new_tokens = []
    for token in tokens:
        if token not in stoi:
            for char in token:
                assert char in stoi, f"token {char} is not in stoi dictionary"
                new_tokens.append(char)
        else:
            new_tokens.append(token)
    return new_tokens


def convert_tokens_to_code(tokens):
    return "".join([token + " " if token == "return" else token for token in tokens])


def wrap_body(simcode_body):
    simcode_body = simcode_body.strip()
    if simcode_body[:4] == "func":
        return simcode_body
    return "func simulation(){" + simcode_body + "}"


def convert_to_int_if_needed(answer):
    if isinstance(answer, int):
        return answer
    if answer.is_integer():
        return int(answer)
    return answer


def execute_simcode(sim_code: str, add_traces=False, precision=2):
    python_code = sim2python(wrap_body(sim_code), add_traces)
    python_code += "print(simulation())"
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    try:
        exec(python_code)
    except:
        raise
    finally:
        sys.stdout = old_stdout
    output = redirected_output.getvalue().split("\n")
    numeric_output = None
    state = output[:-2]
    if output[-2] != "None":
        numeric_output = round(float(output[-2]), 3)
        if numeric_output % 1 == 0:
            numeric_output = int(numeric_output)
        numeric_output = round(convert_to_int_if_needed(numeric_output), precision)
        state.append(f"return {numeric_output}")
    return numeric_output, state


def format_simcode(sim_code: str) -> str:
    sim_code = wrap_body(sim_code)
    chars = InputStream(sim_code)
    lexar = SimCodeLexer(chars)
    tokens = CommonTokenStream(lexar)
    parser = SimCodeParser(tokens)
    tree = parser.parse()
    sim_printer = SimPrinterListener()
    walker = ParseTreeWalker()
    walker.walk(sim_printer, tree)
    return sim_printer.get_generated_code()


def get_vocab(special_tokens=None):
    if special_tokens is None:
        special_tokens = ["<unk>", "<pad>", "<sos>", "<eos>"]
    counter = Counter()
    for i in range(10):
        counter[str(i)] = 1
    for i in range(100):
        counter[f"VAR{i+1}"] = 1
    for token in ["{", "}", "(", ")", ";", "=", "/", "-", "+", "*", ">", "<", "<=", ">=", "!=", "."]:
        counter[token] = 1
    for token in ["repeat", "if", "return"]:
        counter[token] = 1
    return Vocab(counter, min_freq=0, specials=special_tokens, specials_first=True)


def state_trace_exact_match(state, pred_state):
    accuracy = 0
    for state_vars, pred_state_vars in zip(state, pred_state):
        state_vars_split = state_vars.split(",")
        pred_state_vars_split = pred_state_vars.split(",")

        if len(state_vars_split) == len(pred_state_vars_split):
            if all(element in state_vars_split for element in pred_state_vars_split):
                accuracy += 1
    return accuracy / max(len(pred_state), len(state))


def execute_and_compute_state_score(pred_code, target_code):
    """
    try:
        reward = 0.
        answer, state = execute_simcode(target_code, True)
        pred_answer, pred_state = execute_simcode(pred_code, True)

        if answer == pred_answer:
            reward += 1
        else:
            reward -= 1

        memory_state = state[-2]
        memory_pred_state = pred_state[-2]

        state_vars_split = memory_state.split(",")
        pred_state_vars_split = memory_pred_state.split(",")

        if len(state_vars_split) == len(pred_state_vars_split):
            if all(element in state_vars_split for element in pred_state_vars_split):
                reward += 1
            else:
                reward -= 1
        else:
            reward -= 1

        return reward

    except Exception as e:
        return -1000
    """
    try:
        answer, state = execute_simcode(target_code, True)
        pred_answer, pred_state = execute_simcode(pred_code, True)

        # FIXME: Gal's code
        # return state_trace_exact_match(state, pred_state)

        return 1000. * math.isclose(answer, pred_answer, abs_tol=0.001)

    except Exception as e:
        return -1000


def add_state_to_code(state, code):
    variables = state.split(",")
    new_code = variables + [code]
    return ";".join(new_code)


def extract_execution_code_block(code_tokens) -> Tuple[List[List[str]], List[str]]:
    current_state = BlockType.NotInBlock
    blocks_tokens = []
    block_code = []
    current_tokens = []
    current_code_block = ""
    number_of_curly_brackets = 0
    for token in code_tokens:
        # code type
        if token == "if":
            if current_state == BlockType.NotInBlock:
                current_state = BlockType.IfBlock
        elif token == "repeat":
            if current_state == BlockType.NotInBlock:
                current_state = BlockType.IfBlock

        # add information
        current_code_block += token if token != "return" else "return "
        current_tokens.append(token)
        if token == "{":
            number_of_curly_brackets += 1

        # handle finish of code block
        if token == "}":
            if number_of_curly_brackets > 0:
                number_of_curly_brackets -= 1
            if number_of_curly_brackets == 0:
                current_state = BlockType.NotInBlock
                blocks_tokens.append(current_tokens)
                block_code.append(current_code_block)
                current_tokens = []
                current_code_block = ""

        if token == ";" and current_state == BlockType.NotInBlock:
            blocks_tokens.append(current_tokens)
            block_code.append(current_code_block)
            current_tokens = []
            current_code_block = ""
    if current_code_block != "":
        blocks_tokens.append(current_tokens)
        block_code.append(current_code_block)
    return blocks_tokens, block_code
