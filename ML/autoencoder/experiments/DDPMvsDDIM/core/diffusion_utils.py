import math

import torch


def extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape):
    out = values.gather(0, timesteps.long())
    return out.view(timesteps.size(0), *([1] * (len(x_shape) - 1)))


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


def make_beta_schedule(schedule_name: str, timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
    if schedule_name == "linear":
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

    if schedule_name == "cosine":
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5).pow(2)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-5, 0.999)

    raise ValueError(f"Unsupported beta schedule: {schedule_name}")


class DiffusionSchedule:
    def __init__(self, timesteps: int, beta_schedule: str = "linear", beta_start: float = 1e-4, beta_end: float = 2e-2):
        self.timesteps = int(timesteps)
        self.betas = make_beta_schedule(beta_schedule, timesteps, beta_start=beta_start, beta_end=beta_end)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]], dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).clamp(min=1e-20)
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device: torch.device):
        moved = DiffusionSchedule.__new__(DiffusionSchedule)
        moved.timesteps = self.timesteps
        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(moved, name, value.to(device))
        return moved


def q_sample(schedule: DiffusionSchedule, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor = None):
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_alpha = extract(schedule.sqrt_alphas_cumprod, timesteps, x0.shape)
    sqrt_one_minus_alpha = extract(schedule.sqrt_one_minus_alphas_cumprod, timesteps, x0.shape)
    return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise


def predict_x0_from_eps(schedule: DiffusionSchedule, x_t: torch.Tensor, timesteps: torch.Tensor, eps: torch.Tensor):
    return (
        extract(schedule.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t
        - extract(schedule.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) * eps
    )


def predict_x0_from_v(schedule: DiffusionSchedule, x_t: torch.Tensor, timesteps: torch.Tensor, v: torch.Tensor):
    alpha = extract(schedule.sqrt_alphas_cumprod, timesteps, x_t.shape)
    sigma = extract(schedule.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape)
    return alpha * x_t - sigma * v


def predict_eps_from_v(schedule: DiffusionSchedule, x_t: torch.Tensor, timesteps: torch.Tensor, v: torch.Tensor):
    alpha = extract(schedule.sqrt_alphas_cumprod, timesteps, x_t.shape)
    sigma = extract(schedule.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape)
    return sigma * x_t + alpha * v


def build_training_target(
    prediction_target: str,
    schedule: DiffusionSchedule,
    x0: torch.Tensor,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
):
    if prediction_target == "epsilon":
        return noise
    if prediction_target == "v":
        alpha = extract(schedule.sqrt_alphas_cumprod, timesteps, x0.shape)
        sigma = extract(schedule.sqrt_one_minus_alphas_cumprod, timesteps, x0.shape)
        return alpha * noise - sigma * x0
    raise ValueError(f"Unsupported prediction_target: {prediction_target}")


def model_output_to_x0_eps(
    prediction_target: str,
    schedule: DiffusionSchedule,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    model_output: torch.Tensor,
):
    if prediction_target == "epsilon":
        eps = model_output
        x0 = predict_x0_from_eps(schedule, x_t, timesteps, eps)
        return x0, eps
    if prediction_target == "v":
        v = model_output
        x0 = predict_x0_from_v(schedule, x_t, timesteps, v)
        eps = predict_eps_from_v(schedule, x_t, timesteps, v)
        return x0, eps
    raise ValueError(f"Unsupported prediction_target: {prediction_target}")


def compute_snr(schedule: DiffusionSchedule, timesteps: torch.Tensor):
    alpha = schedule.alphas_cumprod.gather(0, timesteps.long())
    return alpha / (1.0 - alpha).clamp(min=1e-8)


def compute_min_snr_weights(
    schedule: DiffusionSchedule,
    timesteps: torch.Tensor,
    prediction_target: str,
    gamma: float,
):
    snr = compute_snr(schedule, timesteps)
    capped = torch.minimum(snr, torch.full_like(snr, gamma))
    if prediction_target == "epsilon":
        return capped / snr.clamp(min=1e-8)
    if prediction_target == "v":
        return capped / (snr + 1.0)
    raise ValueError(f"Unsupported prediction_target: {prediction_target}")


def build_condition_tensor(cond_mode: str, cond_embedding: torch.Tensor, raw_condition: torch.Tensor):
    if cond_mode == "embedding_only":
        return cond_embedding
    if cond_mode == "raw_only":
        return raw_condition
    if cond_mode == "embedding_plus_raw":
        return torch.cat([cond_embedding, raw_condition], dim=1)
    raise ValueError(f"Unsupported cond_mode: {cond_mode}")


@torch.no_grad()
def sample_ddpm(
    model,
    schedule: DiffusionSchedule,
    cond: torch.Tensor,
    latent_dim: int,
    device: torch.device,
    clip_x0: float = 3.0,
    initial_noise: torch.Tensor = None,
    generator: torch.Generator = None,
    prediction_target: str = "epsilon",
):
    batch_size = cond.size(0)
    if initial_noise is None:
        x_t = torch.randn(batch_size, latent_dim, device=device, generator=generator)
    else:
        x_t = initial_noise.to(device).clone()

    for step in reversed(range(schedule.timesteps)):
        t = torch.full((batch_size,), step, device=device, dtype=torch.long)
        model_output = model(x_t, t, cond)
        x0, pred_eps = model_output_to_x0_eps(prediction_target, schedule, x_t, t, model_output)
        x0 = x0.clamp(-clip_x0, clip_x0)
        mean = (
            extract(schedule.posterior_mean_coef1, t, x_t.shape) * x0
            + extract(schedule.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        if step > 0:
            noise = torch.randn(x_t.shape, device=device, dtype=x_t.dtype, generator=generator)
            var = extract(schedule.posterior_variance, t, x_t.shape)
            x_t = mean + torch.sqrt(var) * noise
        else:
            x_t = mean
    return x_t


@torch.no_grad()
def sample_ddim(
    model,
    schedule: DiffusionSchedule,
    cond: torch.Tensor,
    latent_dim: int,
    device: torch.device,
    num_inference_steps: int = 50,
    eta: float = 0.0,
    clip_x0: float = 3.0,
    initial_noise: torch.Tensor = None,
    generator: torch.Generator = None,
    prediction_target: str = "epsilon",
):
    batch_size = cond.size(0)
    if initial_noise is None:
        x_t = torch.randn(batch_size, latent_dim, device=device, generator=generator)
    else:
        x_t = initial_noise.to(device).clone()

    step_indices = torch.linspace(schedule.timesteps - 1, 0, steps=num_inference_steps, device=device)
    step_indices = torch.round(step_indices).long()

    for idx, step in enumerate(step_indices):
        t = torch.full((batch_size,), int(step.item()), device=device, dtype=torch.long)
        model_output = model(x_t, t, cond)
        x0, pred_eps = model_output_to_x0_eps(prediction_target, schedule, x_t, t, model_output)
        x0 = x0.clamp(-clip_x0, clip_x0)

        if idx == len(step_indices) - 1:
            x_t = x0
            continue

        next_step = step_indices[idx + 1]
        alpha_t = schedule.alphas_cumprod[step]
        alpha_prev = schedule.alphas_cumprod[next_step]

        sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
        noise = (
            torch.randn(x_t.shape, device=device, dtype=x_t.dtype, generator=generator)
            if eta > 0
            else torch.zeros_like(x_t)
        )
        direction = torch.sqrt(torch.clamp(1 - alpha_prev - sigma ** 2, min=0.0)) * pred_eps
        x_t = torch.sqrt(alpha_prev) * x0 + direction + sigma * noise

    return x_t
