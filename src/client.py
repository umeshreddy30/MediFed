import flwr as fl
import torch
import os
from model import MedicalCNN
from dataset import load_partitioned_data
from privacy import PrivacyEngine

# Configuration
CLIENT_ID = int(os.environ.get("CLIENT_ID", 0))
TOTAL_CLIENTS = 2
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HospitalClient(fl.client.NumPyClient):
    def __init__(self):
        self.net = MedicalCNN().to(DEVICE)
        self.train_loader = load_partitioned_data(CLIENT_ID, TOTAL_CLIENTS)
        self.privacy = PrivacyEngine(clip_norm=1.5, noise_multiplier=0.02)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def fit(self, parameters, config):
        # 1. Update local model
        self.set_parameters(parameters)
        
        # 2. Local Training
        self.train_local(epochs=1)
        
        # 3. Get updated weights and apply Privacy
        raw_params = [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        safe_params = self.privacy.apply_dp(raw_params)
        
        return safe_params, len(self.train_loader.dataset), {}

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.net.load_state_dict(state_dict, strict=True)

    def train_local(self, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.net.train()
        for _ in range(epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(self.net(images), labels)
                loss.backward()
                optimizer.step()

if __name__ == "__main__":
    # Robust connection logic handles server startup delay
    fl.client.start_numpy_client(
        server_address="server:8080", 
        client=HospitalClient()
    )