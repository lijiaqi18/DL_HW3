import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, ReLU

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import GINConv, GINEConv, DeepGCNLayer, GENConv
from torch_geometric.nn import BatchNorm

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

'''
class DeepGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 16, cached=True, normalize=True)
        self.bn1 = BatchNorm(in_channels)
        self.deep1 = DeepGCNLayer(conv=self.conv1, norm=self.bn1, act=F.relu, dropout=0.5)
        self.conv2 = GCNConv(16, out_channels, cached=True, normalize=True)
        self.bn2 = BatchNorm(16)
        self.deep2 = DeepGCNLayer(conv=self.conv2, norm=self.bn2, act=F.relu, dropout=0.5)

    def forward(self, x, edge_index):
        x = self.deep1(x, edge_index)
        x = self.deep2.conv(x, edge_index, edge_weight)
        x = self.deep2.act(self.deep2.norm(x))
        x = F.dropout(x, p = self.deep2.dropout, training=self.training)
        return x
'''
class DeepGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super().__init__()

        self.node_encoder = Linear(dataset.num_features, hidden_channels)
        # self.edge_encoder = Linear(data.edge_attr.size(-1), hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GCNConv(hidden_channels, hidden_channels, cached=True, normalize=True)
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=self.layers[0].dropout, training=self.training)

        return self.lin(x)

device = torch.device('cpu')
# model = DeepGCN(dataset.num_features, dataset.num_classes).to(device)
model = DeepGCN(hidden_channels=64, num_layers=3).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_weight)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(data):
    model.eval()
    out, accs = model(data.x, data.edge_index, data.edge_weight), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = out[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


for epoch in range(1, 201):
    train(data)
    train_acc, val_acc, test_acc = test(data)
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')