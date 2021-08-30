from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

sf = SmoothingFunction()

from grammer.utils import execute_and_compute_state_score, convert_tokens_to_code


def state_transition_reward(ref_code_tokens, pred_code_tokens) -> float:
    return execute_and_compute_state_score(
        convert_tokens_to_code(ref_code_tokens), convert_tokens_to_code(pred_code_tokens)
    )


def blue_and_same_state_score(ref_code_tokens, pred_code_tokens, gamma) -> float:
    state_score = state_transition_reward(ref_code_tokens, pred_code_tokens)
    bleu = corpus_bleu([[ref_code_tokens]], [pred_code_tokens], smoothing_function=sf.method1, weights=(0.5, 0.5))
    if gamma == 1:
        return bleu
    if state_score < 0:
        return -1000
    return gamma * bleu + (1 - gamma) * state_score
