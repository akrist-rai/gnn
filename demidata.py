import torch
import torch.nn as nn
import torch.nn.functional as F

# --- THE DATA ---
# 3 Nodes, each with 2 features: [Cars, Lanes]
x = torch.tensor([
    [10., 2.],  # Node 0
    [50., 3.],  # Node 1
    [100., 2.]  # Node 2
], dtype=torch.float)

# Connectivity: 0 connects to 1, 1 connects to 2, 2 connects to 0 (Loop)
# Row 0: [0, 1, 1] means Node 0 is connected to Node 1 and Node 2
adjacency_matrix = torch.tensor([
    [0., 1., 0.], # Node 0 listens to Node 1
    [0., 0., 1.], # Node 1 listens to Node 2
    [1., 0., 0.]  # Node 2 listens to Node 0
])
