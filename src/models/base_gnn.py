import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool

class BaselineGNN(nn.Module):
    """
    A standard topological GNN (GCN-based).
    Ignores 3D coordinates. Uses atom types and bond connectivity.
    """
    def __init__(self, hidden_channels=64, num_layers=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.embedding = nn.Embedding(100, hidden_channels) # Atom embedding

        self.convs = nn.ModuleList()
        # 3 Layers of Graph Convolution
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Output Head (Regression)
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        # Unpack
        x, edge_index, batch = data.z, data.edge_index, data.batch
        
        # 1. Initial Embedding
        h = self.embedding(x)

        # 2. Message Passing
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=0.1, training=self.training)

        # 3. Readout (Graph Level Pooling)
        h_graph = global_add_pool(h, batch)

        # 4. Final Prediction
        out = self.lin(h_graph)
        
        return out
