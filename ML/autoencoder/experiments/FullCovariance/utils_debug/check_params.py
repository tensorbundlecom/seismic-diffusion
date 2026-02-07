import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'ML')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'ML/autoencoder')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'ML/autoencoder/experiments/full_cov_cvae')))

from ML.autoencoder.experiments.full_cov_cvae.model_full_cov import FullCovCVAE

model = FullCovCVAE(in_channels=3, latent_dim=128, num_stations=125)
print("Before forward pass:")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

# Mock input
x = torch.randn(1, 3, 129, 111)
mag = torch.zeros(1)
loc = torch.zeros(1, 3)
stat = torch.zeros(1, dtype=torch.long)

model(x, mag, loc, stat)

print("\nAfter forward pass:")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

# Check optimizer creation timing
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
print(f"Optimizer param groups: {len(optimizer.param_groups[0]['params'])}")

# Now check if we add a layer AFTER optimizer
model.encoder.fc_new = torch.nn.Linear(10, 10)
print(f"Optimizer param groups after adding layer: {len(optimizer.param_groups[0]['params'])}")
