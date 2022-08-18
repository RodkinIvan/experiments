from mido import MidiFile
from midi_utils import get_transformed_data, load_transformed
from utils import data_process, standardize

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
    md = MidiFile('/home/ivan/Desktop/Notes/MIDI/1.mid')
    data = get_transformed_data(md)
    # print(data)
    fl_data = data_process(data, n_notes=128)
    print(fl_data.shape)


    # print(standardize(torch.randn(4, 4)[:, 2]))