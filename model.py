import torch
import torch.nn as nn

from functions import *


class LM_RNN(nn.Module):
    def __init__(self, args , output_size, pad_index=0):
        super(LM_RNN, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(output_size, args.embde_size, padding_idx=pad_index)

        # LSTM layer
        self.rnn = nn.LSTM(args.embde_size, args.hidden_size, args.n_layers, bidirectional=args.Bi_directional, batch_first=True)

        # Linear layer to project the hidden layer to our output space 
        # Initially create the linear layer without assigning weights
        self.output = nn.Linear(args.hidden_size, output_size)

        # Dropout layers
        self.drop_embed = VariationalDropout(args.emb_dropout)
        self.drop_out = VariationalDropout(args.out_dropout)

        # Ensure the weights of the output layer are tied with the embedding layer
        # The weight of the Linear layer is set to the transpose of the embedding weight
        self.output.weight = self.embedding.weight

        # Optionally tie biases if you want, typically set to zero or not tied
        #self.output.bias.data.zero_()

    def forward(self, input_sequence):
        # Embedding lookup with dropout
        emb = self.embedding(input_sequence)
        emb_drop = self.drop_embed(emb)

        # RNN
        rnn_out, _ = self.rnn(emb_drop)

        # Apply dropout to RNN output
        rnn_out_drop = self.drop_out(rnn_out)

        # Output layer
        # Permute dimensions because nn.Linear expects (batch, *, features)
        output = self.output(rnn_out_drop).permute(0, 2, 1)

        return output



def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)
