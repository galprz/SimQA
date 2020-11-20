from antlr4 import InputStream, CommonTokenStream

from grammer.code.SimCodeLexer import SimCodeLexer, ParseTreeWalker
from grammer.code.SimCodeParser import SimCodeParser
from grammer.src.SimExecutionBlockListener import SimExecutionBlockListener
from grammer.src.SimTokensListener import Sim2TokensListener
from grammer.tokenizer import SimCodeTokenizer
from grammer.utils import format_simcode, execute_simcode, wrap_body, add_state_to_code, extract_execution_code_block
from reward import checkpoints_reward

if __name__ == "__main__":
    sim_code = """
   func simulation(){
        a=1;
        b=2;
        return a;
    }
    """
    ref_code = """
    func simulation(){
         a=1;
         b=2;
         return a;
     }
     """
    print(add_state_to_code("a=1,b=2", "a=a*b;"))
    tokenizer = SimCodeTokenizer()
    tokens = tokenizer.tokenize(sim_code)
    ref_tokens = tokenizer.tokenize(ref_code)
    print(tokens)
    print(checkpoints_reward(ref_tokens, tokens))
    pass
