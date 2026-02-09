import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
import numpy as np
import obspy
from obspy import read
from scipy import signal
import pandas as pd


class SeismicSTFTDatasetWithMetadata(Dataset):
    """
    PyTorch Dataset for loading seismic waveforms from mseed files and converting them to STFT spectrograms.
    
    Each mseed file contains 3 components (E, N, Z) which are converted to STFT and stacked as 3-channel images.
    Additionally, this dataset includes event metadata (magnitude, location) and station information.
    """
    
    def __init__(
        self,
        data_dir: str = "data/filtered_waveforms",
        event_file: str = "data/events/20140101_20251101_0.0_9.0_9_339.txt",
        channels: list = ["HH", "HN", "EH", "BH"],
        nperseg: int = 256,
        noverlap: int = 192,
        nfft: int = 256,
        normalize: bool = True,
        log_scale: bool = True,
        return_magnitude: bool = True,
        magnitude_col: str = "xM",  # Which magnitude column to use (MD, ML, Mw, Ms, Mb, xM)
        station_list: Optional[List[str]] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to the filtered_waveforms directory
            event_file: Path to the event catalog file
            channels: List of channel types to include (e.g., ["HH", "HN"])
            nperseg: Length of each segment for STFT
            noverlap: Number of points to overlap between segments
            nfft: Length of the FFT used
            normalize: Whether to normalize the spectrograms to [0, 1]
            log_scale: Whether to apply log scaling to the magnitude
            return_magnitude: If True, return magnitude; if False, return complex spectrogram
            magnitude_col: Which magnitude column to use from the event file
            station_list: Optional list of station names to enforce specific mapping
        """
        self.data_dir = Path(data_dir)
        self.event_file = Path(event_file)
        self.channels = channels
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.normalize = normalize
        self.log_scale = log_scale
        self.return_magnitude = return_magnitude
        self.magnitude_col = magnitude_col
        self.station_list_fixed = station_list
        
        # Load event catalog
        print(f"Loading event catalog from {event_file}...")
        self.event_catalog = self._load_event_catalog()
        print(f"Loaded {len(self.event_catalog)} events")
        
        # Calculate location normalization parameters
        self._calculate_location_bounds()
        
        # Collect all mseed files from specified channels
        self.file_paths = []
        for channel in channels:
            channel_dir = self.data_dir / channel
            if channel_dir.exists():
                mseed_files = sorted(channel_dir.glob("*.mseed"))
                self.file_paths.extend(mseed_files)
        
        if len(self.file_paths) == 0:
            # Check if this is OOD or dummy mode where we might want to proceed even without files
            # But normally we raise error.
            # However, if we are in OOD mode, we might point to a different dir.
            # Let's keep the error but be mindful.
            # Changing logic: if strict OOD visualizer runs with 0 files found in training path, it might crash.
            # But here data_dir is passed.
            if not str(data_dir).endswith("ood_waveforms/filtered"):
                 raise ValueError(f"No mseed files found in {self.data_dir} for channels {channels}")
            else:
                 print(f"Warning: No files found in {data_dir}. This might be intentional if just loading for dummy check.")
        
        print(f"Found {len(self.file_paths)} mseed files")
        
        # Build station name list and mapping
        self.station_names, self.station_to_idx = self._build_station_mapping()
        print(f"Found {len(self.station_names)} unique stations")
    
    def _load_event_catalog(self) -> pd.DataFrame:
        """
        Load the event catalog file.
        
        Returns:
            DataFrame with event information
        """
        # Read the event file
        # The file has tab-separated columns with Turkish headers
        # Try different encodings for Turkish characters
        for encoding in ['latin1', 'windows-1254', 'iso-8859-9', 'utf-8']:
            try:
                df = pd.read_csv(
                    self.event_file,
                    sep='\t',
                    encoding=encoding,
                    skipinitialspace=True,
                )
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not decode {self.event_file} with any known encoding")
        
        # Rename columns for easier access
        # Columns: No, Deprem Kodu, Olus tarihi, Olus zamani, Enlem, Boylam, Der(km), xM, MD, ML, Mw, Ms, Mb, Tip, Yer
        df.columns = df.columns.str.strip()
        
        # Create a mapping from event ID (Deprem Kodu) to event information
        # Event ID in filename is like: 20140314175500
        # Event ID in catalog is like: 20140314175500
        df['event_id'] = df['Deprem Kodu'].astype(str).str.strip()
        
        # Convert magnitude columns to float, replacing 0.0 with NaN
        mag_columns = ['xM', 'MD', 'ML', 'Mw', 'Ms', 'Mb']
        for col in mag_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[df[col] == 0.0, col] = np.nan
        
        # Convert location columns to float
        df['latitude'] = pd.to_numeric(df['Enlem'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['Boylam'], errors='coerce')
        df['depth'] = pd.to_numeric(df['Der(km)'], errors='coerce')
        
        return df
    
    def _calculate_location_bounds(self):
        """
        Calculate min/max bounds for event locations for normalization.
        """
        self.lat_min = self.event_catalog['latitude'].min()
        self.lat_max = self.event_catalog['latitude'].max()
        self.lon_min = self.event_catalog['longitude'].min()
        self.lon_max = self.event_catalog['longitude'].max()
        self.depth_min = self.event_catalog['depth'].min()
        self.depth_max = self.event_catalog['depth'].max()
        
        print(f"Location bounds:")
        print(f"  Latitude: [{self.lat_min:.4f}, {self.lat_max:.4f}]")
        print(f"  Longitude: [{self.lon_min:.4f}, {self.lon_max:.4f}]")
        print(f"  Depth: [{self.depth_min:.4f}, {self.depth_max:.4f}] km")
    
    def _build_station_mapping(self) -> Tuple[List[str], Dict[str, int]]:
        """
        Build a list of unique station names from the mseed files.
        
        Returns:
            Tuple of (station_names list, station_to_idx dict)
        """
        station_names = set()
        
        if self.station_list_fixed is not None:
             print(f"Using fixed station list with {len(self.station_list_fixed)} stations.")
             station_names_list = sorted(self.station_list_fixed)
             station_to_idx = {name: idx for idx, name in enumerate(station_names_list)}
             return station_names_list, station_to_idx
        
        for file_path in self.file_paths:
            # Extract station name from filename
            # Filename format: YYYYMMDDHHMMSS_STATION_CHANNEL.mseed
            filename = file_path.stem  # Remove .mseed extension
            parts = filename.split('_')
            if len(parts) >= 2:
                station_name = parts[1]
                station_names.add(station_name)
        
        # Sort for consistency
        station_names_list = sorted(list(station_names))
        station_to_idx = {name: idx for idx, name in enumerate(station_names_list)}
        
        return station_names_list, station_to_idx
    
    def _extract_event_id_from_filename(self, filename: str) -> str:
        """
        Extract event ID from mseed filename.
        
        Args:
            filename: Name of the mseed file (e.g., "20140101214024_EDC_HH.mseed")
        
        Returns:
            Event ID string (e.g., "20140101214024")
        """
        name = Path(filename).stem
        parts = name.split('_')
        if len(parts) >= 3 and parts[0] == "OOD" and (parts[1] == "K" or parts[1] == "POST"):
            return f"{parts[0]}_{parts[1]}_{parts[2]}"
        if len(parts) >= 2 and parts[0] == "OOD":
            # Normalize OOD IDs: OOD_4 or OOD_04 -> OOD_04
            try:
                num = int(parts[1])
                return f"OOD_{num:02d}"
            except ValueError:
                return f"{parts[0]}_{parts[1]}"
        return parts[0]
    
    def _extract_station_from_filename(self, filename: str) -> str:
        """
        Extract station name from mseed filename.
        
        Args:
            filename: Name of the mseed file (e.g., "20140101214024_EDC_HH.mseed")
        
        Returns:
            Station name string (e.g., "EDC")
        """
        # Remove extension
        name = Path(filename).stem
        # Split by underscore and take the second part (station)
        parts = name.split('_')
        if len(parts) >= 2:
            return parts[1]
        return "UNKNOWN"
    
    def _get_event_info(self, event_id: str) -> Optional[Dict]:
        """
        Get event information from the catalog.
        
        Args:
            event_id: Event ID string
        
        Returns:
            Dictionary with event information, or None if not found
        """
        # Find the event in the catalog
        event_rows = self.event_catalog[self.event_catalog['event_id'] == event_id]
        
        if len(event_rows) == 0:
            return None
        
        event = event_rows.iloc[0]
        
        # Get the magnitude (use specified column, fall back to others if NaN)
        magnitude = event[self.magnitude_col]
        if pd.isna(magnitude):
            # Try other magnitude types in order of preference
            for mag_col in ['Mw', 'ML', 'Ms', 'Mb', 'MD', 'xM']:
                if not pd.isna(event[mag_col]):
                    magnitude = event[mag_col]
                    break
        
        # If still no magnitude, use 0.0
        if pd.isna(magnitude):
            magnitude = 0.0
        
        # Normalize location
        lat_norm = (event['latitude'] - self.lat_min) / (self.lat_max - self.lat_min)
        lon_norm = (event['longitude'] - self.lon_min) / (self.lon_max - self.lon_min)
        depth_norm = (event['depth'] - self.depth_min) / (self.depth_max - self.depth_min)
        
        return {
            'magnitude': float(magnitude),
            'latitude': float(event['latitude']),
            'longitude': float(event['longitude']),
            'depth': float(event['depth']),
            'latitude_norm': float(lat_norm),
            'longitude_norm': float(lon_norm),
            'depth_norm': float(depth_norm),
            'location': event['Yer'],
        }
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """
        Load a waveform and convert it to STFT spectrogram, along with event and station metadata.
        
        Returns:
            spectrogram: Tensor of shape (3, freq_bins, time_bins) representing the 3-channel STFT image
            magnitude: Scalar tensor containing event magnitude
            location: Tensor of shape (3,) containing normalized [latitude, longitude, depth]
            station_idx: Integer tensor containing station index
            metadata: Dictionary containing additional information
        """
        file_path = self.file_paths[idx]
        
        try:
            # Read the mseed file
            stream = read(str(file_path))
            
            # Merge traces with the same ID (handles gaps/overlaps)
            stream.merge(fill_value=0)
            
            # Select exactly 3 components (prefer HH, then HN, then others)
            # Group traces by their component (last char of channel: E, N, Z)
            component_groups = {'E': [], 'N': [], 'Z': []}
            for tr in stream:
                comp = tr.stats.channel[-1]
                if comp in component_groups:
                    component_groups[comp].append(tr)
            
            # For each component, pick the best trace based on channel prefix
            final_traces = []
            for comp in ['E', 'N', 'Z']:
                comp_traces = component_groups[comp]
                if not comp_traces:
                    continue
                
                # Sort by channel prefix preference: HH > HN > BH > EH
                pref = {'HH': 0, 'HN': 1, 'BH': 2, 'EH': 3}
                comp_traces.sort(key=lambda x: pref.get(x.stats.channel[:2], 99))
                final_traces.append(comp_traces[0])
            
            stream = obspy.Stream(traces=final_traces)
            
            # Ensure we have exactly 3 components
            if len(stream) != 3:
                # If we don't have exactly 3, maybe it's 1, 2, 3 or some other components?
                # For this project, we strictly need 3.
                raise ValueError(f"Could not find 3 standard components (E,N,Z). Available: {[tr.stats.channel for tr in stream]} in {file_path}")
            
            # Sort traces by channel name to ensure consistent ordering (E, N, Z)
            stream.sort(keys=['channel'])
            
            # Extract event ID and station name from filename
            event_id = self._extract_event_id_from_filename(file_path.name)
            station_name = self._extract_station_from_filename(file_path.name)
            
            # Alternative: get station from stream (should be the same)
            station_from_stream = stream[0].stats.station
            
            # Get event information
            event_info = self._get_event_info(event_id)
            if event_info is None:
                print(f"Warning: Event {event_id} not found in catalog for file {file_path.name}")
                # Use default values
                event_info = {
                    'magnitude': 0.0,
                    'latitude': 0.0,
                    'longitude': 0.0,
                    'depth': 0.0,
                    'latitude_norm': 0.5,
                    'longitude_norm': 0.5,
                    'depth_norm': 0.5,
                    'location': 'UNKNOWN',
                }
            
            # Get station index
            station_idx = self.station_to_idx.get(station_from_stream, 0)
            
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
                    magnitude_spec = np.abs(Zxx)
                    
                    # Apply log scaling
                    if self.log_scale:
                        magnitude_spec = np.log1p(magnitude_spec)  # log(1 + x) to avoid log(0)
                    
                    # Normalize to [0, 1]
                    if self.normalize:
                        mag_min = magnitude_spec.min()
                        mag_max = magnitude_spec.max()
                        if mag_max > mag_min:
                            magnitude_spec = (magnitude_spec - mag_min) / (mag_max - mag_min)
                        else:
                            magnitude_spec = np.zeros_like(magnitude_spec)
                    
                    stft_channels.append(magnitude_spec)
                else:
                    # Return complex spectrogram (real and imaginary parts)
                    stft_channels.append(Zxx)
            
            # Ensure all channels have the same shape before stacking
            if len(stft_channels) > 0:
                min_time_bins = min(c.shape[1] for c in stft_channels)
                stft_channels = [c[:, :min_time_bins] for c in stft_channels]
            
            # Stack the 3 components to create a 3-channel image
            if self.return_magnitude:
                spectrogram = np.stack(stft_channels, axis=0)  # Shape: (3, freq_bins, time_bins)
            else:
                # For complex spectrograms, stack real and imaginary parts
                real_parts = [np.real(c) for c in stft_channels]
                imag_parts = [np.imag(c) for c in stft_channels]
                spectrogram = np.stack(real_parts + imag_parts, axis=0)  # Shape: (6, freq_bins, time_bins)
            
            # Convert to torch tensors
            spectrogram_tensor = torch.from_numpy(spectrogram).float()
            magnitude_tensor = torch.tensor(event_info['magnitude'], dtype=torch.float32)
            location_tensor = torch.tensor([
                event_info['latitude_norm'],
                event_info['longitude_norm'],
                event_info['depth_norm']
            ], dtype=torch.float32)
            station_idx_tensor = torch.tensor(station_idx, dtype=torch.long)
            
            # Create metadata dictionary
            metadata = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'event_id': event_id,
                'station_name': station_from_stream,
                'channel_type': file_path.parent.name,
                'sampling_rate': stream[0].stats.sampling_rate,
                'n_samples': len(stream[0].data),
                'shape': spectrogram_tensor.shape,
                'magnitude': event_info['magnitude'],
                'latitude': event_info['latitude'],
                'longitude': event_info['longitude'],
                'depth': event_info['depth'],
                'location_name': event_info['location'],
                'location_name': event_info['location'],
                'station_idx': station_idx,
                'mag_min': float(mag_min) if self.normalize else 0.0,
                'mag_max': float(mag_max) if self.normalize else 1.0,
            }
            
            return spectrogram_tensor, magnitude_tensor, location_tensor, station_idx_tensor, metadata
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return dummy tensors and error metadata
            dummy_spec_shape = (3 if self.return_magnitude else 6, self.nfft // 2 + 1, 1)
            return (
                torch.zeros(dummy_spec_shape),
                torch.tensor(0.0),
                torch.tensor([0.5, 0.5, 0.5]),
                torch.tensor(0),
                {'error': str(e), 'file_path': str(file_path)}
            )


def collate_fn_with_metadata(batch):
    """
    Custom collate function to handle variable-sized spectrograms with metadata.
    
    This function pads spectrograms to the same size within a batch.
    """
    spectrograms = []
    magnitudes = []
    locations = []
    station_indices = []
    metadata_list = []
    
    # Find the maximum time dimension in the batch
    max_time = max([spec.shape[2] for spec, _, _, _, _ in batch])
    
    for spectrogram, magnitude, location, station_idx, metadata in batch:
        # Skip if there was an error
        if 'error' in metadata:
            continue
        
        # Pad the time dimension to match max_time
        if spectrogram.shape[2] < max_time:
            pad_size = max_time - spectrogram.shape[2]
            spectrogram = torch.nn.functional.pad(spectrogram, (0, pad_size), mode='constant', value=0)
        
        spectrograms.append(spectrogram)
        magnitudes.append(magnitude)
        locations.append(location)
        station_indices.append(station_idx)
        metadata_list.append(metadata)
    
    if len(spectrograms) == 0:
        return None, None, None, None, None
    
    # Stack into batches
    batch_spectrograms = torch.stack(spectrograms, dim=0)
    batch_magnitudes = torch.stack(magnitudes, dim=0)
    batch_locations = torch.stack(locations, dim=0)
    batch_station_indices = torch.stack(station_indices, dim=0)
    
    return batch_spectrograms, batch_magnitudes, batch_locations, batch_station_indices, metadata_list


if __name__ == "__main__":
    # Example usage
    from torch.utils.data import DataLoader
    
    # Create dataset
    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir="../../data/filtered_waveforms",
        event_file="../../data/events/20140101_20251101_0.0_9.0_9_339.txt",
        channels=["HH"],  # Start with just HH channel
        nperseg=256,
        noverlap=192,
        normalize=True,
        log_scale=True,
        magnitude_col="ML",  # Use ML magnitude
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of unique stations: {len(dataset.station_names)}")
    print(f"First 10 stations: {dataset.station_names[:10]}")
    
    # Test loading a single sample
    print("\n" + "="*80)
    print("Loading first sample...")
    print("="*80)
    spectrogram, magnitude, location, station_idx, metadata = dataset[0]
    print(f"\nSpectrogram shape: {spectrogram.shape}")
    print(f"Spectrogram range: [{spectrogram.min():.4f}, {spectrogram.max():.4f}]")
    print(f"\nMagnitude: {magnitude.item():.2f}")
    print(f"Location (normalized): {location.numpy()}")
    print(f"Station index: {station_idx.item()}")
    print(f"Station name: {metadata['station_name']}")
    print(f"\nEvent details:")
    print(f"  Event ID: {metadata['event_id']}")
    print(f"  Magnitude: {metadata['magnitude']:.2f}")
    print(f"  Latitude: {metadata['latitude']:.4f}")
    print(f"  Longitude: {metadata['longitude']:.4f}")
    print(f"  Depth: {metadata['depth']:.2f} km")
    print(f"  Location: {metadata['location_name']}")
    
    # Test with DataLoader
    print("\n" + "="*80)
    print("Testing with DataLoader...")
    print("="*80)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging, increase for faster loading
        collate_fn=collate_fn_with_metadata,
    )
    
    # Load one batch
    batch_specs, batch_mags, batch_locs, batch_stations, batch_metadata = next(iter(dataloader))
    print(f"\nBatch spectrograms shape: {batch_specs.shape}")
    print(f"Batch magnitudes shape: {batch_mags.shape}")
    print(f"Batch locations shape: {batch_locs.shape}")
    print(f"Batch station indices shape: {batch_stations.shape}")
    print(f"\nBatch metadata (first 2 samples):")
    for i, meta in enumerate(batch_metadata[:2]):
        print(f"\n  Sample {i}:")
        print(f"    File: {meta['file_name']}")
        print(f"    Station: {meta['station_name']} (idx: {meta['station_idx']})")
        print(f"    Magnitude: {meta['magnitude']:.2f}")
        print(f"    Location: ({meta['latitude']:.4f}, {meta['longitude']:.4f})")
