# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from nlplingo.oregon.event_models.uoregon.tools.global_constants import *
from nlplingo.oregon.event_models.uoregon.tools.utils import *


class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """

    def __init__(self, in_dim, hidden_dim, num_layers, opt):
        super(GCN, self).__init__()
        self.opt = opt
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.gcn_drop = nn.Dropout(self.opt['gcn_dropout'])

        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(self.num_layers):
            input_dim = self.in_dim if layer == 0 else self.hidden_dim
            self.W.append(nn.Linear(input_dim, self.hidden_dim))

    def forward(self, gcn_inputs, adj):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        pool_mask = (adj.sum(2) + adj.sum(1)).eq(0)

        for l in range(self.num_layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs)  # self loop
            AxW = AxW / denom

            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.num_layers - 1 else gAxW

        return gcn_inputs, pool_mask
