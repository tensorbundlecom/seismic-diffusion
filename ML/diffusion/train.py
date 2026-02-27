from model import DiffusionUNet2D, create_conditioning_vector
import torch
import json
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from diffusers import DDPMScheduler
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import generate, embedding_to_figure, decode_embedding

# --- Config ---
NUM_EPOCHS = 500
BATCH_SIZE = 64
LR = 1e-4
NUM_TRAIN_TIMESTEPS = 1000
LOG_EVERY_N_EPOCHS = 1
CHECKPOINT_EVERY_N_EPOCHS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

writer = SummaryWriter(log_dir="runs/diffusion")

# --- Data ---
embeddings = torch.load("embeddings/embeddings.pt")

# Scale latents to ~unit variance so the N(0,1) noise has the right SNR.
# (Stable Diffusion uses the same trick with its 0.18215 factor.)
EMB_STD = embeddings.std().item()
embeddings = embeddings / EMB_STD
print(f"Embedding std before scaling: {EMB_STD:.5f}  → scaled to: {embeddings.std().item():.5f}")

metadatas = json.load(open("embeddings/metadata.json", "r"))
# Build raw vectors first; keep a copy for use with generate() which normalises internally.
raw_cond_vectors = torch.stack([create_conditioning_vector(m) for m in metadatas])

# --- Normalise only the continuous part (first 4 dims); leave one-hot as-is ---
NUM_CONTINUOUS = 4
cond_mean = raw_cond_vectors[:, :NUM_CONTINUOUS].mean(dim=0)
cond_std  = raw_cond_vectors[:, :NUM_CONTINUOUS].std(dim=0).clamp(min=1e-8)
metadatas_vectors = raw_cond_vectors.clone()
metadatas_vectors[:, :NUM_CONTINUOUS] = (metadatas_vectors[:, :NUM_CONTINUOUS] - cond_mean) / cond_std

print(embeddings.shape)
print(metadatas_vectors.shape)

dataset = TensorDataset(embeddings, metadatas_vectors)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Model & Scheduler ---
model = DiffusionUNet2D(
    in_channels=embeddings.shape[1],
    out_channels=embeddings.shape[1],
    cond_dim=metadatas_vectors.shape[1],
    # base_channels=64 by default → block_out_channels=(64, 128, 256) ≈ 24M params
)
model.model.to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.model.parameters())/1e6:.1f}M")

noise_scheduler = DDPMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS, beta_start=1e-4, beta_end=0.02)
json.dump({
    "emb_std": EMB_STD,
    "cond_mean": cond_mean.tolist(),
    "cond_std": cond_std.tolist(),
}, open("embeddings/scale.json", "w"))  # needed at inference time
optimizer = AdamW(model.model.parameters(), lr=LR, weight_decay=0.0)

# Cosine LR schedule with a short linear warmup
from torch.optim.lr_scheduler import CosineAnnealingLR
WARMUP_EPOCHS = 10
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=LR / 10)

# --- Noise Schedule Visualisation ---
# NUM_VIZ_STEPS = 10
# viz_timesteps = torch.linspace(0, NUM_TRAIN_TIMESTEPS - 1, NUM_VIZ_STEPS, dtype=torch.long)
# sample = embeddings[0].float()
# sample = (sample - sample.min()) / (sample.max() - sample.min()) * 2 - 1
# noise = torch.randn_like(sample)

# noisy_samples = [
#     noise_scheduler.add_noise(sample.unsqueeze(0), noise.unsqueeze(0), t.unsqueeze(0)).squeeze(0)
#     for t in viz_timesteps
# ]

# all_samples = [sample] + noisy_samples
# mean_imgs = [s.mean(dim=0).numpy() for s in all_samples]
# std_imgs  = [s.std(dim=0).numpy()  for s in all_samples]

# # Shared colour limits so differences are actually visible
# mean_vmin, mean_vmax = min(m.min() for m in mean_imgs), max(m.max() for m in mean_imgs)
# std_vmax = max(s.max() for s in std_imgs)

# fig, axes = plt.subplots(2, NUM_VIZ_STEPS + 1, figsize=(2.5 * (NUM_VIZ_STEPS + 1), 5))
# titles = ["Clean\nt=0"] + [f"t={t.item()}" for t in viz_timesteps]
# for i, (title, mean_img, std_img) in enumerate(zip(titles, mean_imgs, std_imgs)):
#     axes[0, i].imshow(mean_img, cmap="seismic", aspect="auto", vmin=mean_vmin, vmax=mean_vmax)
#     axes[0, i].set_title(title); axes[0, i].axis("off")
#     axes[1, i].imshow(std_img,  cmap="hot",     aspect="auto", vmin=0,         vmax=std_vmax)
#     axes[1, i].set_title(title); axes[1, i].axis("off")

# plt.suptitle("Top: mean activation  |  Bottom: std (noise proxy)", fontsize=12)
# plt.tight_layout()
# plt.savefig("noise_schedule_viz.png", dpi=100)
# plt.show()
# print("Saved noise_schedule_viz.png")

# --- Generation helper ---

# Fix a real conditioning vector for reproducible logging.
# Pass the RAW vector — generate() normalises it internally via normalise_cond().
fixed_real_idx = 0
fixed_real_cond = raw_cond_vectors[fixed_real_idx]   # (cond_dim,)  ← raw, not normalised
fixed_rand_idx  = torch.randint(len(raw_cond_vectors), (1,)).item()
fixed_rand_cond = raw_cond_vectors[fixed_rand_idx]    # another real sample for diversity
embedding_shape = tuple(embeddings.shape[1:])          # (C, H, W)

# --- Training Loop ---
for epoch in range(NUM_EPOCHS):
    model.model.train()
    epoch_loss = 0.0

    for batch_embeddings, batch_cond in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
        batch_embeddings = batch_embeddings.to(DEVICE)
        batch_cond = batch_cond.to(DEVICE)

        # Sample random noise
        noise = torch.randn_like(batch_embeddings)

        # Sample random timesteps
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (batch_embeddings.shape[0],), device=DEVICE
        ).long()

        # Add noise to the clean samples (forward diffusion)
        noisy_embeddings = noise_scheduler.add_noise(batch_embeddings, noise, timesteps)

        # Conditioning must be (batch, seq_len, cond_dim) for cross-attention
        cond = batch_cond.unsqueeze(1)

        # Predict the noise
        noise_pred = model.forward(noisy_embeddings, timesteps, cond).sample

        # Compute loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} — Loss: {avg_loss:.6f}  LR: {current_lr:.2e}")
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("LR", current_lr, epoch)

    # Step scheduler (skip warmup epochs)
    if epoch >= WARMUP_EPOCHS:
        scheduler.step()
    else:
        # Linear warmup: scale LR from 0 to LR
        for pg in optimizer.param_groups:
            pg['lr'] = LR * (epoch + 1) / WARMUP_EPOCHS

    if (epoch + 1) % LOG_EVERY_N_EPOCHS == 0:
        # 1. Generate with a real conditioning vector from the dataset
        gen_real = generate(fixed_real_cond, embedding_shape, noise_scheduler, NUM_TRAIN_TIMESTEPS, model, DEVICE)
        decoded_gen_real = decode_embedding(gen_real)  # (C, H, W) tensor
        fig_real = embedding_to_figure(decoded_gen_real, f"Real cond (idx={fixed_real_idx})")
        writer.add_figure("Generation/real_cond", fig_real, global_step=epoch)
        plt.close(fig_real)

        # 2. Generate with a second fixed real conditioning vector for diversity
        gen_rand = generate(fixed_rand_cond, embedding_shape, noise_scheduler, NUM_TRAIN_TIMESTEPS, model, DEVICE)
        decoded_gen_rand = decode_embedding(gen_rand)  # (C, H, W) tensor
        fig_rand = embedding_to_figure(decoded_gen_rand, f"Real cond (idx={fixed_rand_idx})")
        writer.add_figure("Generation/rand_cond", fig_rand, global_step=epoch)
        plt.close(fig_rand)

    if (epoch + 1) % CHECKPOINT_EVERY_N_EPOCHS == 0:
        ckpt_path = f"checkpoints/epoch_{epoch + 1}"
        model.model.save_pretrained(ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")
        # Keep only the last 3 checkpoints to save disk space
        all_ckpts = sorted(
            Path("checkpoints").glob("epoch_*"),
            key=lambda p: int(p.name.split("_")[1])
        )
        for old in all_ckpts[:-3]:
            shutil.rmtree(old)

# --- Save model ---
writer.close()
model.model.save_pretrained("checkpoints/unet2d")
print("Model saved to checkpoints/unet2d")
