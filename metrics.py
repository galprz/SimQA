from abc import ABC, abstractmethod

import numpy as np

from grammer.utils import convert_tokens_to_code, execute_simcode, state_trace_exact_match
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


sf = SmoothingFunction()


class Metric(ABC):
    """
    class represent metric to report
    """

    def __init__(self):
        self.reset()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def update(self, preds, targets):
        raise NotImplementedError()

    @abstractmethod
    def __str__(self):
        raise NotImplementedError()

    @abstractmethod
    def eval(self):
        raise NotImplementedError()


class MSEScore(Metric):
    def __init__(self):
        super().__init__()
        self.accumulated_MSE = 0
        self.errors = np.array([])
        self.reset()

    def reset(self):
        self.errors = np.array([])
        self.accumulated_MSE = 0

    def update(self, preds, targets):
        for pred_seq, tgt_seq in zip(preds, targets):
            pred_code = convert_tokens_to_code(pred_seq)
            target_code = convert_tokens_to_code(tgt_seq)
            try:
                answer, _ = execute_simcode(target_code, True)
                pred_answer, _ = execute_simcode(pred_code, True)
                print("appending "+str((answer-pred_answer)**2))
                np.append(self.errors, [(answer-pred_answer)**2])
            except Exception as e:
                # print(e)
                pass
        self.accumulated_MSE = np.mean(self.errors)

    def eval(self):
        return np.mean(self.errors)

    def __str__(self):
        return f"evaluation is at score: %.4f" % self.eval()


class CorrectAnswersScore(Metric):
    def __init__(self, tgt_vocab):
        super().__init__()
        self._tgt_vocab = tgt_vocab
        self.reset()

    def reset(self):
        self._state_scores = 0
        self._correct_answers = 0
        self._number_of_batches = 0

    def update(self, preds, targets):
        self._number_of_batches += 1
        scores = 0
        correct_answers = 0
        for pred_seq, tgt_seq in zip(preds, targets):
            pred_code = convert_tokens_to_code(pred_seq)
            target_code = convert_tokens_to_code(tgt_seq)
            try:
                answer, state = execute_simcode(target_code, True)
                pred_answer, pred_state = execute_simcode(pred_code, True)
                scores += state_trace_exact_match(state, pred_state)
                correct_answers += 1 if round(float(answer), 6) == round(float(pred_answer), 6) else 0
            except Exception as e:
                pass
        self._state_scores += scores / len(preds)
        self._correct_answers += correct_answers / len(preds)

    def eval(self):
        return (self._correct_answers / self._number_of_batches, self._state_scores / self._number_of_batches)

    def __str__(self):
        return f"Correct answers %.4f, State transitions score: %.4f" % self.eval()


class BleuScore(Metric):
    def __init__(self):
        super().__init__()

    def reset(self):
        self._n_batches = 0
        self._bleu_scores_sum = 0

    def update(self, preds, targets):
        """
        preds need to have dimension of batch_size, number_of_tokens
        targets need to have dimension of batch_size, number_of_tokens
        """
        self._n_batches += 1
        candidates = preds
        refs = [[target] for target in targets]
        bscore = corpus_bleu(refs, candidates, smoothing_function=sf.method1, weights=(0.5, 0.5))
        self._bleu_scores_sum += bscore

    def eval(self):
        return self._bleu_scores_sum / self._n_batches

    def __str__(self):
        return f"Bleu: %.4f" % self.eval()


class BleuAndStateScore(Metric):
    def __init__(self, tgt_vocab, alpha):
        super().__init__()
        self._alpha = alpha
        self._tgt_vocab = tgt_vocab
        self.reset()

    def reset(self):
        self._n_batches = 0
        self._bleu_scores_sum = 0
        self._state_scores = 0
        self._number_of_batches = 0

    def update(self, preds, targets):
        """
        preds need to have dimension of batch_size, number_of_tokens
        targets need to have dimension of batch_size, number_of_tokens
        """
        self._n_batches += 1
        candidates = preds
        refs = [[target] for target in targets]
        bscore = corpus_bleu(refs, candidates, smoothing_function=sf.method1, weights=(0.5, 0.5))
        self._bleu_scores_sum += bscore

        state_scores = 0
        for pred_seq, tgt_seq in zip(preds, targets):
            pred_code = convert_tokens_to_code(pred_seq)
            target_code = convert_tokens_to_code(tgt_seq)
            try:
                _, state = execute_simcode(target_code, True)
                _, pred_state = execute_simcode(pred_code, True)
                state_scores += state_trace_exact_match(state, pred_state)
            except Exception as e:
                pass
        self._state_scores += state_scores / len(preds)

    def eval(self):
        return ((1 - self._alpha) * self._bleu_scores_sum + self._alpha * self._state_scores) / self._n_batches

    def __str__(self):
        return f"Bleu and state score: %.4f" % self.eval()
