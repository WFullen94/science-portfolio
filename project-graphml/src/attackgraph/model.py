"""GraphSAGE link-prediction model.

encode(): learnable per-node embeddings refined by SAGE message passing over the
graph — after 2 layers a group's vector reflects the techniques it uses and the
other groups that use those same techniques (the collaborative-filtering signal).
decode(): score a candidate (source, target) pair by the dot product of their
refined embeddings.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv


class LinkGNN(nn.Module):
    def __init__(self, num_nodes: int, embed_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, embed_dim)
        dims = [embed_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(
            SAGEConv(dims[i], dims[i + 1]) for i in range(num_layers)
        )
        self.dropout = dropout
        nn.init.xavier_uniform_(self.emb.weight)

    def encode(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.emb.weight
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
