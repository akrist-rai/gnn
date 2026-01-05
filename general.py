from torch_geometric.nn import GCNConv

# PyG uses 'edge_index' instead of a full Adjacency Matrix
# Format: [Source Nodes, Target Nodes]
edge_index = torch.tensor([
    [1, 2, 0], # Source (Node sending message)
    [0, 1, 2]  # Target (Node receiving message)
], dtype=torch.long)

class GNNCommon(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # GCNConv handles the "Message Passing" and "Aggregation" internally
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # --- STEP 1 & 2: MESSAGE PASSING + UPDATE ---
        # GCNConv takes node features (x) and connectivity (edge_index)
        # It aggregates neighbors and multiplies by weights in one go
        h = self.conv1(x, edge_index)
        h = F.relu(h)

        # --- STEP 3: READOUT ---
        output = self.lin2(h)
        return output

# Run it
model_common = GNNCommon(input_dim=2, hidden_dim=5, output_dim=1)
output_pyg = model_common(x, edge_index)

print("\nPyG Output (Traffic Scores):")
print(output_pyg.detach().numpy())
