from antlr4 import InputStream, CommonTokenStream

from grammer.code.SimCodeLexer import SimCodeLexer, ParseTreeWalker
from grammer.code.SimCodeParser import SimCodeParser
from grammer.src.SimTokensListener import Sim2TokensListener
from grammer.utils import wrap_body, get_vocab, normalize_sim_code

class SimCodeTokenizer:
    def __init__(self):
        vocab = get_vocab()
        self.stoi = vocab.stoi

    def tokenize(self, sim_code):
        sim_code = wrap_body(normalize_sim_code(sim_code))
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
            if token not in self.stoi :
                for char in token:
                    assert char in self.stoi, f"token {char} is not in stoi dictionary"
                    new_tokens.append(char)
            else:
                new_tokens.append(token)
        return new_tokens

    def detokenize(self, tokens):
        return " ".join([token for token in tokens])
