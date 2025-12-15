# 🏥 Privacy-Preserving Federated Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)

## ⚡ Quick Overview
This project builds a **Distributed AI System** that trains a medical diagnosis model across multiple hospitals without ever sharing patient data. 

It uses **Federated Learning** to train a global model on isolated "edge" devices (simulated by Docker containers), ensuring full privacy compliance (like HIPAA/GDPR) by sharing only mathematical updates, not raw images.

## 🛠️ Tech Stack
* **Framework:** Flower (`flwr`) + PyTorch
* **Infrastructure:** Docker & Docker Compose
* **Privacy:** Differential Privacy (Gradient Clipping & Noise)
* **Data:** MNIST (Proxy for medical X-rays)

## 🚀 How to Run
1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/med-fed-learning.git](https://github.com/YOUR_USERNAME/med-fed-learning.git)
    cd med-fed-learning
    ```

2.  **Start the Network (Server + 2 Hospitals):**
    ```bash
    docker-compose up --build
    ```

3.  **Watch it Learn:**
    View the logs to see the Server aggregate updates from hospitals.
    * **Round 1:** ~85% Accuracy
    * **Round 5:** ~96% Accuracy

## 📂 Project Structure
* `src/server.py` → The central aggregator (Manages the global model).
* `src/client.py` → The hospital node (Trains locally on private data).
* `src/privacy.py` → Adds noise to gradients for security.
* `docker-compose.yml` → Orchestrates the simulated network.
