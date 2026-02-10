import torch
import numpy as np
import os
import sys
import json
from skimage.metrics import structural_similarity as ssim

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

from ML.autoencoder.experiments.General.core.stft_dataset import SeismicSTFTDatasetWithMetadata
from ML.autoencoder.experiments.General.core.model_baseline import ConditionalVariationalAutoencoder
from ML.autoencoder.experiments.FullCovariance.core.model_full_cov import FullCovCVAE
from ML.autoencoder.experiments.NormalizingFlow.core.model_flow import FlowCVAE

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    baseline_chk = "ML/autoencoder/experiments/General/checkpoints/baseline_external_best.pt"
    flow_chk = "ML/autoencoder/experiments/NormalizingFlow/checkpoints/flow_external_best.pt"
    num_stations = 46

    base_model = ConditionalVariationalAutoencoder(in_channels=3, latent_dim=128, num_stations=num_stations).to(device)
    base_model.load_state_dict(torch.load(baseline_chk, map_location=device)['model_state_dict'])
    base_model.eval()

    flow_model = FlowCVAE(in_channels=3, latent_dim=128, num_stations=num_stations, flow_layers=8).to(device)
    flow_model.load_state_dict(torch.load(flow_chk, map_location=device)['model_state_dict'])
    flow_model.eval()

    with open("data/station_list_external_full.json", 'r') as f:
        station_list = json.load(f)

    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir="data/ood_waveforms/post_training/filtered",
        event_file="data/events/ood_catalog_post_training.txt",
        channels=['HH'],
        magnitude_col='xM',
        station_list=station_list
    )

    results = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            spec, mag, loc, station_idx, meta = dataset[i]
            if 'error' in meta: continue
            
            spec_in = spec.unsqueeze(0).to(device)
            mag_in = mag.unsqueeze(0).to(device)
            loc_in = loc.unsqueeze(0).to(device)
            sta_in = station_idx.unsqueeze(0).to(device)
            
            r_base, _, _ = base_model(spec_in, mag_in, loc_in, sta_in)
            r_flow, _, _, _, _ = flow_model(spec_in, mag_in, loc_in, sta_in)
            
            orig = spec_in[0, 2].cpu().numpy()
            base = r_base[0, 2].cpu().numpy()
            flow = r_flow[0, 2].cpu().numpy()
            
            def get_ssim(s_target, s_pred):
                s1 = (s_target - np.min(s_target)) / (np.max(s_target) - np.min(s_target) + 1e-8)
                s2 = (s_pred - np.min(s_pred)) / (np.max(s_pred) - np.min(s_pred) + 1e-8)
                return ssim(s1, s2, data_range=1.0)
            
            ssim_base = get_ssim(orig, base)
            ssim_flow = get_ssim(orig, flow)
            
            results.append({
                'event': meta['event_id'],
                'station': meta['station_name'],
                'base': ssim_base,
                'flow': ssim_flow
            })

    print(f"{'Event':<15} | {'Station':<8} | {'Base SSIM':<10} | {'Flow SSIM':<10} | {'Winner':<6}")
    print("-" * 60)
    for r in results:
        winner = "Base" if r['base'] > r['flow'] else "Flow"
        print(f"{r['event']:<15} | {r['station']:<8} | {r['base']:<10.4f} | {r['flow']:<10.4f} | {winner:<6}")

if __name__ == "__main__":
    main()
