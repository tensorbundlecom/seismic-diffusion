import os
import glob
from pathlib import Path
from typing import Iterable, Optional, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
from obspy import read
from scipy import signal
from tqdm import tqdm


class SeismicSTFTDataset(Dataset):
    """
    PyTorch Dataset for loading seismic waveforms from mseed files and converting them to STFT spectrograms.
    
    Each mseed file contains 3 components (E, N, Z) which are converted to STFT and stacked as 3-channel images.
    """
    
    def __init__(
        self,
        data_dir: str = "data/filtered_waveforms",
        channels: list = ["HH", "HN", "EH", "BH"],
        nperseg: int = 256,
        noverlap: int = 192,
        nfft: int = 256,
        normalize: bool = True,
        log_scale: bool = True,
        return_magnitude: bool = True,
        target_freq_bins: int = None,
        target_time_bins: int = None,
        resample_hz: float = 100.0,
        target_seconds: float = 70.0,
        global_normalization: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Path to the filtered_waveforms directory
            channels: List of channel types to include (e.g., ["HH", "HN"])
            nperseg: Length of each segment for STFT
            noverlap: Number of points to overlap between segments
            nfft: Length of the FFT used
            normalize: Whether to normalize the spectrograms to [0, 1]
            log_scale: Whether to apply log scaling to the magnitude
            return_magnitude: If True, return magnitude; if False, return complex spectrogram
            target_freq_bins: If set, bilinearly resize the STFT frequency axis to this size.
            target_time_bins: If set, bilinearly resize the STFT time axis to this size.
            resample_hz: Resample every trace to this sampling rate before the STFT.
                This makes the physical analysis window (nperseg/fs seconds) and the
                frequency axis (0..fs/2) identical across channel types, instead of
                each rate producing a differently "stretched" spectrogram. None = off.
            target_seconds: After resampling, trim/zero-pad each trace to exactly
                round(resample_hz * target_seconds) samples so the STFT shape is
                uniform with no distortion. None = leave native length.
            global_normalization: If True, normalize every magnitude spectrogram
                using one log-domain min/max fitted on the training indices. Call
                :meth:`fit_global_normalization` or
                :meth:`set_global_normalization_stats` before reading samples.
        """
        self.data_dir = Path(data_dir)
        self.channels = channels
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.normalize = normalize
        self.log_scale = log_scale
        self.return_magnitude = return_magnitude
        self.target_freq_bins = int(target_freq_bins) if target_freq_bins else None
        self.target_time_bins = int(target_time_bins) if target_time_bins else None
        self.resample_hz = float(resample_hz) if resample_hz else None
        self.target_seconds = float(target_seconds) if target_seconds else None
        self.global_normalization = bool(global_normalization)
        self.global_min: Optional[float] = None
        self.global_max: Optional[float] = None
        self.target_samples = (
            int(round(self.resample_hz * self.target_seconds))
            if (self.resample_hz and self.target_seconds)
            else None
        )
        
        # Collect all mseed files from specified channels
        self.file_paths = []
        for channel in channels:
            channel_dir = self.data_dir / channel
            if channel_dir.exists():
                mseed_files = sorted(channel_dir.glob("*.mseed"))
                self.file_paths.extend(mseed_files)
        
        if len(self.file_paths) == 0:
            raise ValueError(f"No mseed files found in {self.data_dir} for channels {channels}")
        
        print(f"Found {len(self.file_paths)} mseed files")
    
    def __len__(self):
        return len(self.file_paths)

    def _prep_trace_data(self, trace):
        """Resample to resample_hz and trim/zero-pad to target_samples; return float32 data."""
        if self.resample_hz is not None and abs(trace.stats.sampling_rate - self.resample_hz) > 1e-6:
            trace.resample(self.resample_hz)
        data = trace.data.astype(np.float32)
        if self.target_samples is not None:
            n = self.target_samples
            if data.shape[0] >= n:
                data = data[:n]
            else:
                data = np.pad(data, (0, n - data.shape[0]), mode="constant")
        return data

    def _resize_spectrogram(self, spectrogram_tensor):
        """Bilinearly resize (C, F, T) to the configured target (F, T), if any."""
        if self.target_freq_bins is None and self.target_time_bins is None:
            return spectrogram_tensor
        f = self.target_freq_bins if self.target_freq_bins is not None else spectrogram_tensor.shape[1]
        t = self.target_time_bins if self.target_time_bins is not None else spectrogram_tensor.shape[2]
        if (spectrogram_tensor.shape[1], spectrogram_tensor.shape[2]) == (f, t):
            return spectrogram_tensor
        resized = torch.nn.functional.interpolate(
            spectrogram_tensor.unsqueeze(0), size=(f, t), mode="bilinear", align_corners=False
        )
        return resized.squeeze(0)

    @property
    def global_normalization_stats(self) -> Optional[dict]:
        """Return fitted global normalization statistics in config-friendly form."""
        if self.global_min is None or self.global_max is None:
            return None
        return {"min": self.global_min, "max": self.global_max}

    def set_global_normalization_stats(self, global_min: float, global_max: float) -> Tuple[float, float]:
        """Set reusable log-domain global normalization bounds.

        The values must be finite and satisfy ``global_max >= global_min``. A
        zero range is valid and maps all values to zero during normalization.
        """
        global_min = float(global_min)
        global_max = float(global_max)
        if not np.isfinite(global_min) or not np.isfinite(global_max):
            raise ValueError("Global normalization bounds must be finite.")
        if global_max < global_min:
            raise ValueError(
                "Global normalization requires global_max >= global_min, "
                f"got {global_max} < {global_min}."
            )
        self.global_min = global_min
        self.global_max = global_max
        return self.global_min, self.global_max

    def _global_normalization_is_ready(self) -> bool:
        """Whether global normalization has usable fitted bounds."""
        return self.global_min is not None and self.global_max is not None

    def _apply_global_normalization(self, spectrogram_tensor: torch.Tensor) -> torch.Tensor:
        """Normalize a post-resize log-magnitude tensor without clipping."""
        if not self._global_normalization_is_ready():
            raise RuntimeError(
                "Global normalization is enabled but statistics are not fitted. "
                "Call fit_global_normalization(training_indices) or "
                "set_global_normalization_stats(global_min, global_max) first."
            )
        assert self.global_min is not None and self.global_max is not None
        value_range = self.global_max - self.global_min
        if value_range == 0.0:
            return torch.zeros_like(spectrogram_tensor)
        return (spectrogram_tensor - self.global_min) / value_range

    def _make_spectrogram_tensor(
        self, file_path: Path, per_sample_normalize: bool
    ) -> Tuple[torch.Tensor, object]:
        """Load one file and build its log-domain, optionally resized STFT tensor.

        ``per_sample_normalize`` is intentionally separate from ``self.normalize``
        so fitting can inspect unnormalized, post-resize log magnitudes.
        """
        stream = read(str(file_path))
        if len(stream) != 3:
            raise ValueError(f"Expected 3 traces, got {len(stream)} in {file_path}")
        stream.sort(keys=['channel'])

        stft_channels = []
        for trace in stream:
            data = self._prep_trace_data(trace)
            _, _, Zxx = signal.stft(
                data,
                fs=trace.stats.sampling_rate,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                nfft=self.nfft,
                return_onesided=True,
                boundary='zeros',
                padded=True,
            )
            if self.return_magnitude:
                magnitude = np.abs(Zxx)
                if self.log_scale:
                    magnitude = np.log1p(magnitude)
                if per_sample_normalize:
                    mag_min = magnitude.min()
                    mag_max = magnitude.max()
                    if mag_max > mag_min:
                        magnitude = (magnitude - mag_min) / (mag_max - mag_min)
                    else:
                        magnitude = np.zeros_like(magnitude)
                stft_channels.append(magnitude)
            else:
                stft_channels.append(Zxx)

        if self.return_magnitude:
            spectrogram = np.stack(stft_channels, axis=0)
        else:
            real_parts = [np.real(component) for component in stft_channels]
            imag_parts = [np.imag(component) for component in stft_channels]
            spectrogram = np.stack(real_parts + imag_parts, axis=0)
        return self._resize_spectrogram(torch.from_numpy(spectrogram).float()), stream

    def fit_global_normalization(
        self, indices: Iterable[int], show_progress: bool = True
    ) -> Tuple[float, float]:
        """Fit one scalar log-domain min/max over valid training samples only.

        Each selected event's three components are transformed through log scaling
        and final resizing before their values contribute. Files that fail to load
        (or contain no finite values) are skipped. The fitted tuple can be saved
        and later restored with :meth:`set_global_normalization_stats`.
        """
        if not self.return_magnitude:
            raise ValueError("Global normalization is only supported for magnitude spectrograms.")
        if not self.log_scale:
            raise ValueError("Global normalization requires log_scale=True for log-domain bounds.")

        selected_indices = [int(idx) for idx in indices]
        fitted_min = np.inf
        fitted_max = -np.inf
        valid_samples = 0
        with tqdm(
            selected_indices,
            total=len(selected_indices),
            desc="Fitting global normalization",
            unit="file",
            dynamic_ncols=True,
            disable=not show_progress,
        ) as progress:
            for idx in progress:
                file_path = None
                try:
                    file_path = self.file_paths[int(idx)]
                    spectrogram_tensor, _ = self._make_spectrogram_tensor(
                        file_path, per_sample_normalize=False
                    )
                    finite_values = spectrogram_tensor[torch.isfinite(spectrogram_tensor)]
                    if finite_values.numel() == 0:
                        continue
                    fitted_min = min(fitted_min, float(finite_values.min().item()))
                    fitted_max = max(fitted_max, float(finite_values.max().item()))
                    valid_samples += 1
                except Exception as error:
                    sample_label = str(file_path) if file_path is not None else f"index {idx!r}"
                    tqdm.write(f"Skipping {sample_label} while fitting global normalization: {error}")

        if valid_samples == 0:
            raise RuntimeError("Could not fit global normalization: no valid selected samples were available.")
        return self.set_global_normalization_stats(fitted_min, fitted_max)

    def __getitem__(self, idx):
        """
        Load a waveform and convert it to STFT spectrogram.
        
        Returns:
            spectrogram: Tensor of shape (3, freq_bins, time_bins) representing the 3-channel STFT image
            metadata: Dictionary containing file path and other information
        """
        file_path = self.file_paths[idx]
        if self.global_normalization and not self.return_magnitude:
            raise ValueError("Global normalization is only supported for magnitude spectrograms.")
        if self.global_normalization and not self.log_scale:
            raise ValueError("Global normalization requires log_scale=True for log-domain bounds.")
        if self.global_normalization and not self._global_normalization_is_ready():
            raise RuntimeError(
                "Global normalization is enabled but statistics are not fitted. "
                "Call fit_global_normalization(training_indices) or "
                "set_global_normalization_stats(global_min, global_max) first."
            )
        
        try:
            spectrogram_tensor, stream = self._make_spectrogram_tensor(
                file_path,
                per_sample_normalize=self.normalize and not self.global_normalization,
            )
            if self.global_normalization:
                spectrogram_tensor = self._apply_global_normalization(spectrogram_tensor)

            # Create metadata dictionary
            metadata = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'channel_type': file_path.parent.name,
                'sampling_rate': stream[0].stats.sampling_rate,
                'n_samples': len(stream[0].data),
                'shape': spectrogram_tensor.shape,
            }
            
            return spectrogram_tensor, metadata
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return a zero tensor and error metadata
            dummy_shape = (3 if self.return_magnitude else 6, self.nfft // 2 + 1, 1)
            return torch.zeros(dummy_shape), {'error': str(e), 'file_path': str(file_path)}


def collate_fn(batch):
    """
    Custom collate function to handle variable-sized spectrograms.
    
    This function pads spectrograms to the same size within a batch.
    """
    spectrograms = []
    metadata_list = []
    
    # Find the maximum time dimension in the batch
    max_time = max([spec.shape[2] for spec, _ in batch])
    
    for spectrogram, metadata in batch:
        # Skip if there was an error
        if 'error' in metadata:
            continue
            
        # Pad the time dimension to match max_time
        if spectrogram.shape[2] < max_time:
            pad_size = max_time - spectrogram.shape[2]
            spectrogram = torch.nn.functional.pad(spectrogram, (0, pad_size), mode='constant', value=0)
        
        spectrograms.append(spectrogram)
        metadata_list.append(metadata)
    
    if len(spectrograms) == 0:
        return None, None
    
    # Stack into a batch
    batch_spectrograms = torch.stack(spectrograms, dim=0)
    
    return batch_spectrograms, metadata_list


if __name__ == "__main__":
    # Example usage
    from torch.utils.data import DataLoader
    
    # Create dataset
    dataset = SeismicSTFTDataset(
        data_dir="../../data/filtered_waveforms",
        channels=["HH"],  # Start with just HH channel
        nperseg=256,
        noverlap=192,
        normalize=True,
        log_scale=True,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test loading a single sample
    print("\nLoading first sample...")
    spectrogram, metadata = dataset[0]
    print(f"Spectrogram shape: {spectrogram.shape}")
    print(f"Spectrogram range: [{spectrogram.min():.4f}, {spectrogram.max():.4f}]")
    print(f"Metadata: {metadata}")
    
    # Test with DataLoader
    print("\nTesting with DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging, increase for faster loading
        collate_fn=collate_fn,
    )
    
    # Load one batch
    batch_specs, batch_metadata = next(iter(dataloader))
    print(f"Batch shape: {batch_specs.shape}")
    print(f"Batch metadata (first 2):")
    for i, meta in enumerate(batch_metadata[:2]):
        print(f"  Sample {i}: {meta['file_name']}, shape={meta['shape']}")
