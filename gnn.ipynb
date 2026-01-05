
class GNNFromScratch(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # The Neural Network part (Weights)
        # We start with input_dim (2 features) and go to hidden_dim (5 features)
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adjacency_matrix):
        # --- STEP 1: MESSAGE PASSING & AGGREGATION ---
        # Matrix Multiplication (A @ x) does the "Summing Neighbors" automatically
        # If Node 0 is connected to Node 1, it grabs Node 1's features here.
        neighbor_messages = torch.matmul(adjacency_matrix, x)

        # Optional: Add self-loops (The node should also consider its own features)
        # We simply add the original 'x' to the aggregated messages
        aggregated = neighbor_messages + x

        # --- STEP 2: UPDATE (The Neural Network) ---
        # Pass the aggregated info through the Dense Layer
        h = self.lin1(aggregated)
        h = F.relu(h) # Activation

        # --- STEP 3: READOUT (Output Layer) ---
        # 5 dim -> 1 dim prediction
        output = self.lin2(h)
        return output

# Run it
model_scratch = GNNFromScratch(input_dim=2, hidden_dim=5, output_dim=1)
output = model_scratch(x, adjacency_matrix)

print("Scratch Output (Traffic Scores):")
print(output.detach().numpy())
