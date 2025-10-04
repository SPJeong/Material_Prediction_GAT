##### model.py (GAT)

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GATConv


class GAT_Model(nn.Module):
    def __init__(self, in_channels=57, out_channels=1024, descriptor_ecfp2_size=1217, dropout=0.2):
        super().__init__()
        # GAT
        self.conv1 = GATConv(in_channels, in_channels * 2, heads=5, dropout=dropout)
        self.conv2 = GATConv(in_channels * 2 * 5, in_channels * 3, heads=5, dropout=dropout)
        self.conv3 = GATConv(in_channels * 3 * 5, in_channels * 2, heads=5, dropout=dropout)
        self.conv4 = GATConv(in_channels * 2 * 5, in_channels * 2 * 5, heads=1, dropout=dropout)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # GAT last layer
        self.gat_fc = nn.Linear(in_channels * 2 * 5, out_channels)

        # final FC
        self.final_fc1 = nn.Linear(out_channels + descriptor_ecfp2_size, 1024)  # (1024+1217) -> 1024
        self.final_fc2 = nn.Linear(1024, 2048)
        self.final_fc3 = nn.Linear(2048, 1024)
        self.final_fc4 = nn.Linear(1024, 256)
        self.final_fc5 = nn.Linear(256, 1)

    def forward(self, data_graph, data_descriptor_ECFP):
        x, edge_index, batch = data_graph.x, data_graph.edge_index, data_graph.batch

        # GAT forward
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = self.conv4(x, edge_index)
        x = self.relu(x)

        # Global pooling
        x = torch_geometric.nn.global_add_pool(x, batch)

        # GAT fc
        x = self.gat_fc(x)

        # concat GAT + descriptor_ECFP
        combined_features = torch.cat([x, data_descriptor_ECFP], dim=1)

        # final FC
        out = self.relu(self.final_fc1(combined_features))
        out = self.dropout(out)
        out = self.relu(self.final_fc2(out))
        out = self.dropout(out)
        out = self.relu(self.final_fc3(out))
        out = self.dropout(out)
        out = self.relu(self.final_fc4(out))
        out = self.final_fc5(out)

        return out


def count_parameters(model):
    """
    Calculates the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Calculate and print the total parameters
if __name__ == '__main__':
    my_deep_model = GAT_Model()
    total_params = count_parameters(my_deep_model)
    print(f"Total trainable parameters: {total_params:,}")