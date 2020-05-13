#!/usr/bin/env/python
# This module contains the GGNN model definition.
# The code was originally copied from
# https://github.com/thaonguyen19/gated-graph-neural-network-pytorch
# However, the original code was faulty in several significant ways. Most importantly, there was a
# mistake in the way messages were passed along the edges (in what is now compute_embeddings()).
# These bugs have been changed in the current version. See README.md for more details.

import torch as tt
import torch.nn as nn


class AlignModel(nn.Module):

    def __init__(self, out_dim, n_nodes):
        super(AlignModel, self).__init__()
        self.out_dim = out_dim
        self.n_nodes = n_nodes
        self.similarity = nn.Sequential(nn.Linear(self.out_dim * 2, 1), nn.Sigmoid())

    def compute_embeddings(self, prop_state, A):
        return input

    def forward(self, A, prop_state):
        embeds = self.compute_embeddings(prop_state, A)

        sents = embeds.shape[1] // 2

        original = embeds[:, :sents, :].unsqueeze(1).repeat(1, sents, 1, 1)
        simplified = embeds[:, sents:, :].unsqueeze(2).repeat((1, 1, sents, 1))
        embed_matrix = tt.cat((original, simplified), 3)  # BATCH X SENTS X SENTS X 2 * EMBEDS

        return self.similarity(embed_matrix).squeeze()


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propagator(nn.Module):
    """
    Gated Propagator for GGNN
    Using GRU gating mechanism
    """
    def __init__(self, state_dim, n_nodes, n_edge_types):
        super(Propagator, self).__init__()

        self.n_nodes = n_nodes
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )
        self.state_dim = state_dim

    def forward(self, state_in, state_out, state_cur, A):  # A = [A_in, A_out]

        # state_cur             BATCH, NODE,        EMBED
        # state_out, state_in   BATCH, NODE * EDGE, EMBED
        # A                     BATCH, NODE,        NODE * EDGE * 2
        # A_in, A_out           BATCH, NODE,        NODE * EDGE

        A_in = A[:, :, :self.n_nodes*self.n_edge_types]
        A_out = A[:, :, self.n_nodes*self.n_edge_types:]

        a_in = tt.bmm(A_in, state_in)  # batch size x |V| x state dim
        a_out = tt.bmm(A_out, state_out)
        a = tt.cat((a_in, a_out, state_cur), 2)  # batch size x |V| x 3*state dim

        r = self.reset_gate(a.view(-1, self.state_dim*3))  # batch size*|V| x state_dim
        z = self.update_gate(a.view(-1, self.state_dim*3))
        r = r.view(-1, self.n_nodes, self.state_dim)
        z = z.view(-1, self.n_nodes, self.state_dim)
        joined_input = tt.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.transform(joined_input.view(-1, self.state_dim*3))
        h_hat = h_hat.view(-1, self.n_nodes, self.state_dim)
        output = (1 - z) * state_cur + z * h_hat
        return output


class GGNN(AlignModel):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, state_dim, annotation_dim, n_edge_types, n_nodes, n_steps):
        super(GGNN, self).__init__(state_dim, n_nodes)

        assert (state_dim >= annotation_dim, 'state_dim must be no less than annotation_dim')

        self.annotation_dim = annotation_dim
        self.n_edge_types = n_edge_types
        self.n_steps = n_steps
        self.state_dim = state_dim
        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propagation Model
        self.propagator = Propagator(self.state_dim, self.n_nodes, self.n_edge_types)

        self.out = nn.Linear(self.state_dim, self.state_dim)
        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def compute_embeddings(self,  prop_state, A):
        """
        What was the forward call in original code
        :param prop_state:  Initial node states.       [BATCH, NODES, EMBEDDING_SIZE]
        :param A:           Adjacency matrix.          [BATCH, NODES, NODES * EDGE_TYPES * 2]
        :param A:           [ANCHOR_ID, POS_ID, NEG_ID]
        :return:
        """
        batch, graphs, _, _ = prop_state.shape
        prop_state = prop_state.view(batch * graphs, self.n_nodes, -1)
        A = A.view(batch * graphs, self.n_nodes, -1)

        # PROP state is initialized to Annotation somewhere before
        for i_step in range(self.n_steps):
            # print ("PROP STATE SIZE:", prop_state.size()) #batch size x |V| x state dim
            in_states = []
            out_states = []

            # in_fcs[i] -> in_fcs.__getitem__(i) ->
            # self.in_{i}, which is a linear layer state_dim -> state_dim
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            in_states = tt.stack(in_states).transpose(0, 1).contiguous()  # Batch, edge, node, embed
            in_states = in_states.view(-1, self.n_nodes*self.n_edge_types, self.state_dim)
            out_states = tt.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_nodes*self.n_edge_types, self.state_dim) # batch size x |V||E| x state dim

            prop_state = self.propagator(in_states, out_states, prop_state, A)

        return tt.sum(self.out(prop_state), dim=1).view((batch, graphs, self.state_dim))


class AlignLSTM(AlignModel):

    def __init__(self, state_dim, n_nodes, n_layers, n_hidden, n_output, drop_p=0.5):
        super(AlignLSTM, self).__init__(n_output, n_nodes)
        self.n_layers = n_layers  # number of LSTM layers
        self.n_hidden = n_hidden  # number of hidden nodes in LSTM
        self.n_output = n_output

        self.lstm = nn.LSTM(state_dim, n_hidden, n_layers, batch_first=True, dropout=drop_p)
        self.dropout = nn.Dropout(drop_p)
        self.fc = nn.Linear(n_hidden, n_output)
        self.sigmoid = nn.Sigmoid()

    def compute_embeddings(self, prop_state, A):
        """
        What was the forward call in original code
        :param prop_state:  Initial node states.       [BATCH, NODES, EMBEDDING_SIZE]
        :param A:           Adjacency matrix.          [BATCH, NODES, NODES * EDGE_TYPES * 2]
        :param A:           [ANCHOR_ID, POS_ID, NEG_ID]
        :return:
        """
        batch, graphs, _, _ = prop_state.shape
        prop_state = prop_state.view(batch * graphs, self.n_nodes, -1)

        lstm_out, _ = self.lstm(prop_state)  # (batch_size, seq_length, n_hidden)
        lstm_out = self.dropout(lstm_out)
        fc_out = self.fc(lstm_out)  # (batch_size, seq_length, n_output)
        sigmoid_out = self.sigmoid(fc_out)  # (batch_size, seq_length, n_output)
        # extract the output of ONLY the LAST output of the LAST element of the sequence
        sigmoid_last = sigmoid_out[:, -1, :]  # (batch_size, 1)

        return sigmoid_last.view((batch, graphs, self.n_output))