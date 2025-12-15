import flwr as fl
import torch
import os
from model import MedicalCNN
from dataset import load_partitioned_data
from typing import Dict, Optional, Tuple

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = MedicalCNN()
        # Server holds the "Global Test Set" for unbiased evaluation
        self.test_loader = load_partitioned_data(client_id=-1, total_clients=0)

    def aggregate_fit(self, server_round, results, failures):
        # 1. Standard FedAvg Aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Round {server_round} aggregation complete. Saving model...")
            
            # 2. Convert Flower Parameters to PyTorch State Dict
            params_dict = zip(self.net.state_dict().keys(), fl.common.parameters_to_ndarrays(aggregated_parameters))
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.net.load_state_dict(state_dict, strict=True)
            
            # 3. Save Model Checkpoint
            os.makedirs("models", exist_ok=True)
            torch.save(self.net.state_dict(), f"models/model_round_{server_round}.pth")
            
            # 4. Evaluate Global Model on Server Test Set
            acc, loss = self.evaluate_global()
            print(f"Global Model Accuracy: {acc*100:.2f}% | Loss: {loss:.4f}")

        return aggregated_parameters, aggregated_metrics

    def evaluate_global(self):
        self.net.eval()
        correct, total, loss_sum = 0, 0, 0.0
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self.net(images)
                loss_sum += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total, loss_sum / len(self.test_loader)

if __name__ == "__main__":
    strategy = SaveModelStrategy(
        fraction_fit=1.0, 
        min_fit_clients=2,
        min_available_clients=2
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )