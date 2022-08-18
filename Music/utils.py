import torch
from torch import Tensor
from torch.nn.functional import one_hot

from typing import Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_batch(source: Tensor, bptt: int, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


def standardize(array: Tensor):
    return (array - torch.min(array))/(torch.max(array)-torch.min(array))


def data_process(data, n_notes) -> Tensor:
    size = data.shape[0]
    """Converts raw text into a flat Tensor."""
    data = torch.cat([
        one_hot(torch.tensor(data['note']), num_classes=n_notes),
        torch.tensor(data['time']).reshape(size, 1), 
        torch.tensor(data['dur']).reshape(size, 1),
    ], dim=1).float()

    data[:, n_notes] = standardize(data[:, n_notes])
    data[:, n_notes+1] = standardize(data[:, n_notes+1])
    return data


def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len, *data.shape[1:]).transpose(0, 1).contiguous()
    return data.to(device)