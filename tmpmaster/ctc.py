from typing import List

import torch
import numpy as np


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
