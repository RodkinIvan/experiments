import torch.nn
import torch.nn as nn
from torch.nn import Transformer


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()

        self.toQ = nn.Linear()
        pass

    def forward(self, x):
        pass



class Autobot(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 num_encoder_layers,
                 num_decoder_layers,
                 dim_feedforward,
                 dropout=0.1,
                 activation='relu'
                 ):
        super(Autobot, self).__init__()
        pass

    def forward(self, x):
        pass


