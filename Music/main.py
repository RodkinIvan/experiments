from mido import MidiFile
from midi_utils import get_transformed_data, load_transformed
from utils import batchify, data_process, get_batch, device

from midi_transformer import MidiTransformer

import torch
import torch.nn as nn
from torch import Tensor

import math
import time

n_notes = 128
n_real_features = 2
batch_size = 20
n_epochs = 200

trans_conf = dict(
    n_notes=n_notes,
    n_real_features=n_real_features,
    d_model=200,
    nhead=4,
    d_hid=200,
    nlayers=2,
    dropout=0.3
)

model = MidiTransformer(**trans_conf).to(device)

criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def train(model, dataset, bptt, epoch):
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 1
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(dataset) // bptt
    for batch, i in enumerate(range(0, dataset.size(0) - 1, bptt)):
        data, targets = get_batch(dataset, i, bptt)

        seq_len = data.size(0)
        if seq_len != bptt:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, n_notes+n_real_features), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()
        

if __name__ == '__main__':

    md = MidiFile('/home/ivan/Desktop/Notes/MIDI/1.mid')
    data = get_transformed_data(md)
    train_data = data_process(data, n_notes=n_notes)
    train_data = batchify(train_data, batch_size)
    for epoch in range(n_epochs):
        train(model, train_data, 10, epoch)