from grammer.utils import *


if __name__ == "__main__":
    sim_code = """
       func simulation(){
            a=1;
            b=1;
            c=16//2;
            repeat(c){
                a = a*b;
                b = b+1;
            }
            return a;
        }
        """
    print(format_simcode(sim_code))

    normal_form = normalize_sim_code(sim_code)
    print(format_simcode(normal_form))

    vocab = get_vocab(["<pad>"])
    tokenizer(normal_form, vocab.stoi)

    print(sim2python(sim_code, False))

    print(execute_simcode(sim_code, True))

    sim_code_a = """
       func simulation(){
            a=1;
            b=1;
            c=16//2;
            repeat(c){
                a = a*b;
                b = b+1;
            }
            return a;
        }
        """
    sim_code_b = """
       func simulation(){
            a=1;
            b=1;
            c=16//2;
            repeat(c){
                a = a*b;
                b = b+1;
            }
            return a;
        }
        """

    _, state_trace_a = execute_simcode(sim_code_a, True)
    _, state_trace_b = execute_simcode(sim_code_b, True)

    state_trace_exact_match(state_trace_a, state_trace_b)
