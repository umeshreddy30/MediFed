import torch
import numpy as np
from typing import List

class PrivacyEngine:
    def __init__(self, clip_norm: float = 1.0, noise_multiplier: float = 0.05):
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier

    def apply_dp(self, params: List[np.ndarray]) -> List[np.ndarray]:
        noisy_params = []
        for p in params:
            # 1. Convert to tensor for manipulation
            tensor_p = torch.tensor(p)
            
            # 2. Gradient Clipping (L2 Norm)
            # This ensures no single data point impacts the model too much
            norm = torch.norm(tensor_p)
            scale = min(1.0, self.clip_norm / (norm + 1e-6))
            clipped_p = tensor_p * scale
            
            # 3. Add Gaussian Noise
            noise = torch.normal(mean=0.0, std=self.noise_multiplier, size=clipped_p.shape)
            final_p = clipped_p + noise
            
            noisy_params.append(final_p.numpy())
            
        return noisy_params