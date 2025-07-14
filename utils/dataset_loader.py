from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

def load_dataset(name="Cora"):
    dataset = Planetoid(root=f'data/{name}', name=name, transform=T.NormalizeFeatures())
    data = dataset[0]  # Graph object
    return dataset, data