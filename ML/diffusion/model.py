from diffusers import UNet2DConditionModel
import torch

# Define a wrapper for the UNet2DModel with cross-attention conditioning
class DiffusionUNet2D:
    def __init__(self, in_channels, out_channels, cond_dim, base_channels=64):
        self.model = UNet2DConditionModel(
            sample_size=None,
            in_channels=in_channels,
            out_channels=out_channels,
            # 3 resolution levels: (64, 128, 256)
            # The first block preserves the in_channels count so there
            # is no information bottleneck at the very start.
            block_out_channels=(
                base_channels,
                base_channels * 2,
                base_channels * 4,
            ),
            layers_per_block=2,
            down_block_types=(
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
            ),
            up_block_types=(
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=cond_dim,
        )

    def forward(self, x, timesteps, cond):
        return self.model(x, timestep=timesteps, encoder_hidden_states=cond)
    
NUM_STATIONS = 46  # station_idx runs 0 … 45

def create_conditioning_vector(metadata):
    """Returns a conditioning vector of shape (4 + NUM_STATIONS,).
    The first 4 dims are continuous (magnitude, lat, lon, depth);
    the remaining NUM_STATIONS dims are a one-hot encoding of the station.
    """
    continuous = torch.tensor(
        [metadata['magnitude'], metadata['latitude'],
         metadata['longitude'], metadata['depth']],
        dtype=torch.float32,
    )
    one_hot = torch.zeros(NUM_STATIONS, dtype=torch.float32)
    one_hot[int(metadata['station_idx'])] = 1.0
    return torch.cat([continuous, one_hot])  # (4 + NUM_STATIONS,)