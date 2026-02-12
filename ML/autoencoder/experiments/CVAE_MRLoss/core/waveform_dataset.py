import os
import sys
from typing import List

import numpy as np
import obspy
import torch
from torch.utils.data import Dataset

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

from ML.autoencoder.experiments.General.core.stft_dataset import SeismicSTFTDatasetWithMetadata


class SeismicSTFTDatasetWithWaveform(Dataset):
    """
    Wrapper around existing STFT dataset that adds Z-component waveform tensors.
    This keeps experiment changes isolated from shared core code.
    """

    def __init__(self, target_fs=100.0, target_len=7300, **dataset_kwargs):
        self.base = SeismicSTFTDatasetWithMetadata(**dataset_kwargs)
        self.target_fs = float(target_fs)
        self.target_len = int(target_len)

    def __len__(self):
        return len(self.base)

    def _read_waveform(self, file_path: str):
        st = obspy.read(file_path)
        st.merge(fill_value=0)

        # Keep alignment with training/evaluation convention.
        st.resample(self.target_fs)
        tr = st.select(component='Z')[0] if st.select(component='Z') else st[0]
        wav = tr.data.astype(np.float32)

        if len(wav) > self.target_len:
            wav = wav[: self.target_len]
        elif len(wav) < self.target_len:
            wav = np.pad(wav, (0, self.target_len - len(wav)))

        return wav

    def __getitem__(self, idx):
        spec, mag, loc, station_idx, meta = self.base[idx]

        if 'error' in meta:
            return spec, mag, loc, station_idx, torch.zeros(self.target_len), meta

        try:
            wav = self._read_waveform(meta['file_path'])
            wav_t = torch.from_numpy(wav).float()
            return spec, mag, loc, station_idx, wav_t, meta
        except Exception as exc:
            err_meta = dict(meta)
            err_meta['error'] = f'waveform_read_error: {exc}'
            return spec, mag, loc, station_idx, torch.zeros(self.target_len), err_meta


def collate_fn_with_waveform(batch):
    specs = []
    mags = []
    locs = []
    stations = []
    waveforms = []
    metas = []

    valid = [item for item in batch if 'error' not in item[5]]
    if len(valid) == 0:
        return None, None, None, None, None, None

    max_time = max(item[0].shape[2] for item in valid)
    max_wlen = max(item[4].shape[0] for item in valid)

    for spec, mag, loc, station_idx, wav, meta in valid:
        if spec.shape[2] < max_time:
            spec = torch.nn.functional.pad(spec, (0, max_time - spec.shape[2]), mode='constant', value=0)

        if wav.shape[0] < max_wlen:
            wav = torch.nn.functional.pad(wav, (0, max_wlen - wav.shape[0]), mode='constant', value=0)

        specs.append(spec)
        mags.append(mag)
        locs.append(loc)
        stations.append(station_idx)
        waveforms.append(wav)
        metas.append(meta)

    return (
        torch.stack(specs, dim=0),
        torch.stack(mags, dim=0),
        torch.stack(locs, dim=0),
        torch.stack(stations, dim=0),
        torch.stack(waveforms, dim=0),
        metas,
    )
