import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGPerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss for spectrogram-like 3-channel inputs.

    This compares intermediate VGG feature maps between reconstructed and
    original spectrograms and avoids waveform/iSTFT conversion entirely.
    """

    def __init__(
        self,
        use_imagenet_weights: bool = True,
        resize_to_224: bool = True,
        device: str = "cpu",
    ):
        super().__init__()

        try:
            from torchvision.models import vgg16, VGG16_Weights
        except ImportError as exc:
            raise RuntimeError(
                "VGG perceptual loss requested, but torchvision is not installed."
            ) from exc

        weights = VGG16_Weights.IMAGENET1K_V1 if use_imagenet_weights else None
        vgg_features = vgg16(weights=weights).features.eval()
        vgg_features.requires_grad_(False)

        # Common VGG blocks after relu1_2, relu2_2, relu3_3, relu4_3
        block_end_indices = [3, 8, 15, 22]
        blocks = []
        start_idx = 0
        for end_idx in block_end_indices:
            blocks.append(vgg_features[start_idx : end_idx + 1])
            start_idx = end_idx + 1
        self.blocks = nn.ModuleList(blocks)
        self.blocks.eval()

        self.resize_to_224 = resize_to_224

        # ImageNet normalization used by VGG pretrained weights.
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.to(device)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        # Expected shape: (B, C, H, W)
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got shape {tuple(x.shape)}")

        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) > 3:
            x = x[:, :3]
        elif x.size(1) == 2:
            x = torch.cat([x, x[:, :1]], dim=1)

        x = x.clamp(0.0, 1.0)
        if self.resize_to_224 and x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        x = (x - mean) / std
        return x

    def _forward_blocks(self, x: torch.Tensor):
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        return feats

    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        rec = self._prepare_input(reconstructed)
        org = self._prepare_input(original)

        with torch.no_grad():
            target_features = self._forward_blocks(org)

        pred_features = self._forward_blocks(rec)
        loss = 0.0
        for pred, target in zip(pred_features, target_features):
            loss = loss + F.l1_loss(pred, target)
        return loss


class PhaseNetPerceptualLoss(nn.Module):
    """
    Optional PhaseNet-based perceptual loss for seismic spectrogram reconstructions.

    This module turns spectrograms into proxy waveforms and compares PhaseNet outputs
    between original and reconstructed inputs. The PhaseNet parameters are frozen.
    """

    def __init__(
        self,
        pretrained: str = "stead",
        nperseg: int = 256,
        noverlap: int = 192,
        nfft: int = 256,
        device: str = "cpu",
        eps: float = 1e-6,
    ):
        super().__init__()

        try:
            import seisbench.models as sbm
        except ImportError as exc:
            raise RuntimeError(
                "PhaseNet perceptual loss requested, but 'seisbench' is not installed."
            ) from exc

        self.phasenet = sbm.PhaseNet.from_pretrained(pretrained)
        self.phasenet.eval()
        self.phasenet.requires_grad_(False)
        self.phasenet.to(device)

        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.eps = eps

        hop_length = self.nperseg - self.noverlap
        if hop_length <= 0:
            raise ValueError(
                f"Invalid STFT params: nperseg={self.nperseg}, noverlap={self.noverlap}. "
                "Expected noverlap < nperseg."
            )
        self.hop_length = hop_length

    @staticmethod
    def _extract_first_tensor(output):
        """Recursively extract a tensor from nested model outputs."""
        if torch.is_tensor(output):
            return output
        if isinstance(output, (list, tuple)):
            for item in output:
                tensor = PhaseNetPerceptualLoss._extract_first_tensor(item)
                if tensor is not None:
                    return tensor
            return None
        if isinstance(output, dict):
            for value in output.values():
                tensor = PhaseNetPerceptualLoss._extract_first_tensor(value)
                if tensor is not None:
                    return tensor
            return None
        return None

    def _spectrogram_to_proxy_waveform(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Convert log-magnitude spectrograms to proxy waveforms.

        Uses zero-phase iSTFT when possible and falls back to temporal envelopes if iSTFT
        cannot be applied for shape reasons.
        """
        # spectrogram: (B, C, F, T)
        linear_mag = torch.expm1(spectrogram.clamp_min(0.0))
        batch_size, channels, freq_bins, time_bins = linear_mag.shape

        expected_freq_bins = self.nfft // 2 + 1
        if freq_bins != expected_freq_bins:
            # Fallback: keep temporal structure by pooling over frequencies.
            waveform = linear_mag.mean(dim=2)
        else:
            flat_mag = linear_mag.reshape(batch_size * channels, freq_bins, time_bins)
            complex_spec = torch.complex(flat_mag, torch.zeros_like(flat_mag))

            window = torch.hann_window(
                self.nperseg,
                device=spectrogram.device,
                dtype=spectrogram.dtype,
            )

            try:
                flat_wave = torch.istft(
                    complex_spec,
                    n_fft=self.nfft,
                    hop_length=self.hop_length,
                    win_length=self.nperseg,
                    window=window,
                    center=True,
                    onesided=True,
                    return_complex=False,
                )
                waveform = flat_wave.reshape(batch_size, channels, -1)
            except RuntimeError:
                # Shape mismatch or backend issue: fallback to temporal envelopes.
                waveform = linear_mag.mean(dim=2)

        # Per-channel normalization stabilizes PhaseNet activations.
        mean = waveform.mean(dim=-1, keepdim=True)
        std = waveform.std(dim=-1, keepdim=True).clamp_min(self.eps)
        waveform = (waveform - mean) / std
        return waveform

    def _phasenet_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Run frozen PhaseNet and return a tensor suitable for feature matching."""
        output = self.phasenet(waveform)
        tensor = self._extract_first_tensor(output)
        if tensor is None:
            raise RuntimeError("Could not extract tensor output from PhaseNet forward pass.")
        return tensor

    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """
        Compute PhaseNet perceptual loss between reconstructed and original spectrograms.
        """
        rec_wave = self._spectrogram_to_proxy_waveform(reconstructed)
        org_wave = self._spectrogram_to_proxy_waveform(original)

        with torch.no_grad():
            target_features = self._phasenet_features(org_wave)

        pred_features = self._phasenet_features(rec_wave)
        return F.l1_loss(pred_features, target_features)
