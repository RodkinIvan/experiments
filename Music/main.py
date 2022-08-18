from mido import MidiFile
from midi_utils import get_transformed_data, load_transformed

from midi_transformer import MidiTransformer

import torch
from torch import Tensor


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


if __name__ == '__main__':
    model = MidiTransformer(
        n_notes=128,
        n_real_features=2,
        d_model=200,
        nhead=4,
        d_hid=200,
        nlayers=2,
        dropout=0.3
    )
    