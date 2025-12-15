import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

def load_partitioned_data(client_id: int, total_clients: int, batch_size=32):
    """
    Simulates Non-IID data. 
    Client 0 gets mostly classes 0-4.
    Client 1 gets mostly classes 5-9.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Check if we are server (needs test set) or client (needs train set)
    is_server = (client_id == -1)
    
    dataset = datasets.MNIST('./data', train=not is_server, download=True, transform=transform)
    
    if is_server:
        # Server gets the validation set
        return DataLoader(dataset, batch_size=1000)

    # CLIENT LOGIC: Non-IID Split
    targets = np.array(dataset.targets)
    
    # Simple Non-IID: Sort data by label, then slice.
    # This means Client 0 gets digits 0-4, Client 1 gets 5-9
    # In a real scenario, this extreme bias causes "Catastrophic Forgetting"
    # which FL algorithms aim to solve.
    idxs = np.argsort(targets)
    
    split_size = len(dataset) // total_clients
    start = client_id * split_size
    end = (client_id + 1) * split_size
    
    client_indices = idxs[start:end]
    
    loader = DataLoader(Subset(dataset, client_indices), batch_size=batch_size, shuffle=True)
    print(f"[Client {client_id}] Loaded {len(client_indices)} samples.")
    return loader