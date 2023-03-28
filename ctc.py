from typing import Union, List
import math
import collections

import torch
import numpy as np

ListOfStrings = List[str]
ListOfInts = List[int]

ListOfIntsOrstrings = Union[ListOfInts, ListOfStrings]


class CTCDecoder:
    @staticmethod
    def decode(
        sequence: ListOfIntsOrstrings, blank_label: Union[str, int]
    ) -> ListOfIntsOrstrings:
        if not len(sequence):
            return []
        token = sequence[0]
        i = 1
        decoded_seq = [token]
        while i < len(sequence):
            while sequence[i] == token and i < len(sequence) - 1:
                i += 1
            token = sequence[i]
            decoded_seq.append(token)
            i += 1

        return [d for d in decoded_seq if d != blank_label]

    @classmethod
    def __call__(
        cls,
        sequence: ListOfIntsOrstrings,
        blank_label: Union[str, int],
        tokenizer=None,
    ):
        decoded = cls.decode(sequence, blank_label)
        if tokenizer is not None:
            decoded = tokenizer.decode(decoded)
        return decoded


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, blank=0):
        super().__init__()
        self.blank = blank

    def forward(self, indices: torch.Tensor, attention=None) -> List[str]:
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        if attention is not None:
            indices = indices[attention > 0]

        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i.item() for i in indices if i != self.blank]
        return indices
