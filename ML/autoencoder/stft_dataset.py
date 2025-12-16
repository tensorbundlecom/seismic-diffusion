import os
import glob
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from obspy import read
from scipy import signal


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
        """
        self.data_dir = Path(data_dir)
        self.channels = channels
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.normalize = normalize
        self.log_scale = log_scale
        self.return_magnitude = return_magnitude
        
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
    
    def __getitem__(self, idx):
        """
        Load a waveform and convert it to STFT spectrogram.
        
        Returns:
            spectrogram: Tensor of shape (3, freq_bins, time_bins) representing the 3-channel STFT image
            metadata: Dictionary containing file path and other information
        """
        file_path = self.file_paths[idx]
        
        try:
            # Read the mseed file
            stream = read(str(file_path))
            
            # Ensure we have 3 components
            if len(stream) != 3:
                raise ValueError(f"Expected 3 traces, got {len(stream)} in {file_path}")
            
            # Sort traces by channel name to ensure consistent ordering (E, N, Z)
            stream.sort(keys=['channel'])
            
            # Initialize list to store STFT for each component
            stft_channels = []
            
            for trace in stream:
                # Get the waveform data
                data = trace.data.astype(np.float32)
                
                # Compute STFT
                f, t, Zxx = signal.stft(
                    data,
                    fs=trace.stats.sampling_rate,
                    nperseg=self.nperseg,
                    noverlap=self.noverlap,
                    nfft=self.nfft,
                    return_onesided=True,
                    boundary='zeros',
                    padded=True
                )
                
                # Get magnitude spectrogram
                if self.return_magnitude:
                    magnitude = np.abs(Zxx)
                    
                    # Apply log scaling
                    if self.log_scale:
                        magnitude = np.log1p(magnitude)  # log(1 + x) to avoid log(0)
                    
                    # Normalize to [0, 1]
                    if self.normalize:
                        mag_min = magnitude.min()
                        mag_max = magnitude.max()
                        if mag_max > mag_min:
                            magnitude = (magnitude - mag_min) / (mag_max - mag_min)
                        else:
                            magnitude = np.zeros_like(magnitude)
                    
                    stft_channels.append(magnitude)
                else:
                    # Return complex spectrogram (real and imaginary parts)
                    stft_channels.append(Zxx)
            
            # Stack the 3 components to create a 3-channel image
            if self.return_magnitude:
                # print(stft_channels[0].shape, stft_channels[1].shape, stft_channels[2].shape)
                spectrogram = np.stack(stft_channels, axis=0)  # Shape: (3, freq_bins, time_bins)
            else:
                # For complex spectrograms, stack real and imaginary parts
                real_parts = [np.real(c) for c in stft_channels]
                imag_parts = [np.imag(c) for c in stft_channels]
                spectrogram = np.stack(real_parts + imag_parts, axis=0)  # Shape: (6, freq_bins, time_bins)
            
            # Convert to torch tensor
            spectrogram_tensor = torch.from_numpy(spectrogram).float()
            
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
