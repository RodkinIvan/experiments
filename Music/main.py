from mido import MidiFile
from midi_utils import get_transformed_data, load_transformed
from utils import batchify, data_process, get_batch, device

from midi_transformer import MidiTransformer

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import one_hot

import math
import time

import pandas as pd
import numpy as np

n_notes = 128
n_real_features = 2
batch_size = 20
n_epochs = 200
bptt = 40

generate_n = 50

trans_conf = dict(
    n_notes=n_notes,
    n_real_features=n_real_features,
    d_model=200,
    nhead=2,
    d_hid=200,
    nlayers=4,
    dropout=0.1
)

model = MidiTransformer(**trans_conf).to(device)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def train(model, dataset, bptt, epoch):


    criterion = nn.CrossEntropyLoss()
    lr = 5.0  # learning rate
    optimizer = torch.optim.Adam(model.parameters())

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
            # lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

def generate(model: nn.Module, start_notes: Tensor, length: int):
    model.eval()
    cur_seq = start_notes

    src_mask = generate_square_subsequent_mask(len(cur_seq)).to(device)
    for i in range(length-1):
        data = cur_seq.view(len(cur_seq), 1, start_notes.shape[1]).to(device)
        output = model(data, src_mask)[-1, 0, :]

        output[:-2] = one_hot(
            torch.tensor(
                np.random.choice(
                    list(range(output.shape[0]-2)),
                    p=output[:-2].softmax(dim=0).cpu().detach().numpy()
                )
            ),
            num_classes=output.shape[0]-2
        )
        output[-2:] = output[-2:].clip(0, 1)
        output = output.to(device)
        cur_seq = torch.cat([cur_seq, output.view(1, start_notes.shape[1])])
        src_mask = generate_square_subsequent_mask(len(cur_seq)).to(device)
    notes = cur_seq[:, :-2].argmax(dim=1).view(-1, 1)
    times = cur_seq[:, -2:]

    return torch.cat([notes, times], dim=1).cpu().detach().numpy()

if __name__ == '__main__':
    md = MidiFile('/home/ivan/Desktop/Notes/MIDI/1.mid')
    # data = get_transformed_data(md)
    data = pd.read_csv('/home/ivan/DataspellProjects/experiments/Music/all_midies.csv')


    train_data = data_process(data, n_notes=n_notes)
    train_data = batchify(train_data, batch_size)
    for epoch in range(1,n_epochs+1):
        train(model, train_data, bptt, epoch)
    sequence = generate(model, train_data[0][10:11], generate_n)
    df_gen = pd.DataFrame(sequence, columns=['note', 'time', 'dur'])

    df_gen['velocity'] = 80
    df_gen[['time','dur']] *= 500

    df_gen = df_gen.astype('int32')
    torch.save(model, 'cur_model.pt')

    load_transformed(df_gen, '/home/ivan/Desktop/Notes/MIDI/tr1.mid')
