import collections

import torch

from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


def normalize_answer(s):
    """Lower text and remove extra whitespace."""

    def white_space_fix(text):
        return " ".join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(lower(s))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_pred, a_gold):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_pred, a_gold):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


class StringMetricBase(Metric):
    def __init__(self, output_transform=lambda x: x, device="cpu"):
        self._num_correct = None
        self._num_examples = None
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._num_correct = torch.tensor(0, device=self._device)
        self._num_examples = 0
        super().reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output

        num_examples = len(y)
        self._num_examples += num_examples

        for a_gold, a_pred in zip(y, y_pred):
            self._num_correct += torch.tensor(
                self._metric(a_gold, a_pred), device=self._device
            )

    @sync_all_reduce("_num_examples", "_num_correct:SUM")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                "CustomAccuracy must have at least one example before it can be computed."
            )
        return self._num_correct.item() / self._num_examples

    def _metric(self, y_pred, y):
        raise NotImplementedError


class ExactMatch(StringMetricBase):
    def _metric(self, y_pred, y):
        return compute_exact(y_pred, y)


class WordF1(StringMetricBase):
    def _metric(self, y_pred, y):
        return compute_f1(y_pred, y)
