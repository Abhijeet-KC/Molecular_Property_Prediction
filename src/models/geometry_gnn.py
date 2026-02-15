import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool

class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift

class GaussianSmearing(nn.Module):
    """
    Expands distances into a vector of radial basis functions (RBF).
    This allows the network to learn complex dependencies on distance.
    """
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / ((stop - start) / num_gaussians) ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class InteractionBlock(MessagePassing):
    """
    Continuous-filter convolution block (cf. SchNet).
    The filter is generated from the edge distances.
    """
    def __init__(self, hidden_channels, num_gaussians):
        super().__init__(aggr='add') # additive aggregation
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, hidden_channels),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        # edge_attr here is the RBF expanded distance
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j: features of neighbor j
        # edge_attr: RBF distance features
        
        # Filter generation: W = MLP(d_ij)
        W = self.mlp(edge_attr)
        
        # Continuous convolution: x_j * W
        return x_j * W

    def update(self, aggr_out, x):
        # Residual connection
        return x + self.lin(aggr_out)



class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift

class GeometricGNN(nn.Module):
    """
    A Geometry-Aware Graph Neural Network (similar to SchNet).
    It uses atomic numbers and 3D positions to predict properties.
    """
    def __init__(self, hidden_channels=128, num_filters=128, num_interactions=3, num_gaussians=50, cutoff=10.0):
        super().__init__()
        self.cutoff = cutoff
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions

        # 1. Atom Embedding: Z -> vector
        self.embedding = nn.Embedding(100, hidden_channels) # Up to element 100

        # 2. Distance Expansion
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        # 3. Interaction Blocks
        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians)
            self.interactions.append(block)

        # 4. Output Blocks (Decoder)
        self.atom_decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels // 2, 1)
        )



    def _get_edge_features(self, pos, edge_index):
        # Calculate distances
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        
        # Expand distances with RBF
        edge_attr = self.distance_expansion(dist)
        return edge_attr

    def forward(self, data):
        # Unpack data
        z, pos, batch = data.z, data.pos, data.batch
        # Note: QM9 uses 'z' for atomic numbers, 'x' usually features.
        # If 'z' is not present, we might need data.x (if using one-hot). 
        # QM9 has data.z.

        # If we don't have edge_index computed based on cutoff radius (dynamic graph),
        # we can compute it here using `radius_graph`.
        # However, QM9 comes with `edge_index` based on bonds.
        # For geometric DL, we often ignore chemical bonds and use spatial proximity.
        from torch_geometric.nn import radius_graph
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        
        edge_attr = self._get_edge_features(pos, edge_index)

        # Initialize node features
        h = self.embedding(z)

        # Interaction Phase
        for interaction in self.interactions:
            h = interaction(h, edge_index, edge_attr)

        # Readout Phase
        # We predict a value for valid atoms and sum them up
        atom_out = self.atom_decoder(h)
        graph_out = global_add_pool(atom_out, batch)
        
        return graph_out

