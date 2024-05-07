
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Dict, Optional

def generate_padding_mask(sequences: torch.Tensor, padding_idx: int) -> torch.BoolTensor:
    '''
        sequences: (bs, seq_len, dim)
    '''
    if sequences is None:
        return None

    if len(sequences.shape) == 2: # (bs, seq_len)
        __seq = sequences.unsqueeze(dim=-1) # (bs, seq_len, 1)
    else:
        __seq = sequences

    mask = (torch.sum(__seq, dim=-1) == (padding_idx*__seq.shape[-1])).long() * -10e4 # (b_s, seq_len)
    return mask.unsqueeze(1).unsqueeze(1) # (bs, 1, 1, seq_len)