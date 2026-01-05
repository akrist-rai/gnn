multiplying the Adjacency Matrix by the Feature Matrix (A×X) automatically performs the "Sum Aggregation" step you described. It sums the vectors of all neighbors for every node simultaneously.


    Memory: The "Scratch" version creates an N×N matrix. If you have 1 million nodes (Google Maps), this crashes your RAM. The "Common" version only stores the list of existing roads (Edges), which is much smaller.

    Aggregation Logic:

    Scratch: We manually did A @ x + x (Sum Aggregation + Self Loop).

    Common (GCNConv): It automatically adds self-loops and normalizes the sum (usually using Mean or a specific spectral normalization) so nodes with 100 neighbors don't explode with huge values compared to nodes with 1 neighbor.
