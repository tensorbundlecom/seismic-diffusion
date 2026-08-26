"""
First-order seismogram characteristics on the test split (GWM paper, Fig. 3).

For every held-out sample (the checkpoint's validation split) this compares, per magnitude
bin, the distribution (mean ± std) of:
  - time-domain log-amplitude envelopes, and
  - Fourier amplitude spectrum log-amplitudes
between the real seismograms and diffusion synthetics generated with the *same*
conditioning vector (magnitude, distance, azimuth, depth, snr, [vs30], station,
channel). Pairing the conditioning makes the marginal comparison meaningful:
both populations share the same joint distribution of everything except the
waveform itself.

Both real and synthetic waveforms are converted to ground acceleration in
m/s^2. Counts-domain checkpoints use station response removal (FDSN
level="response", see fetch_station_responses.py); physical-acceleration
checkpoints are already in that unit and bypass it. The synthetic pipeline is
diffusion sampling -> VAE decode -> Griffin-Lim -> domain-aware conversion.
Legacy per-event-normalized AEs also
apply the historical AmplitudeMLP gain and waveform rescale; globally
normalized AEs recover magnitude directly and bypass that model.

Run from the project root:
    python eval/evaluate_first_order.py compute [--steps 1000] [--batch_size 64] [--limit N]
    python eval/evaluate_first_order.py plot    [--bins 1,1.5,2,2.5,3,3.5,4.5,6]
    python eval/evaluate_first_order.py all     [...]

`compute` is resumable: per-sample features (and generated waveforms in their
recorded domain) are cached in eval/first_order/ and only missing samples are
processed on re-runs. `plot` only needs the cache, so bins can be changed
without regenerating.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DIFF_DIR = ROOT / "ML" / "diffusion"
sys.path.insert(0, str(ROOT))

from ML.amplitude.model import AmplitudeMLP           # noqa: E402
from ML.autoencoder.inference import load_model        # noqa: E402
from ML.diffusion.model import (                        # noqa: E402
    DiffusionUNet2D,
    create_conditioning_vector,
)
from ML.diffusion.reconstruction import (               # noqa: E402
    ReconstructionSpec,
    checkpoint_stft_config,
    decoded_to_magnitude,
    diffusion_cache_tag,
    postprocess_griffinlim_waveform,
    resolve_reconstruction_spec,
)
from ML.diffusion.waveform_domain import (               # noqa: E402
    INSTRUMENT_COUNTS,
    physical_acceleration_to_motion,
)
from embedding_artifacts import (  # noqa: E402
    artifact_path,
    deterministic_fraction_subset,
    deterministic_selection_tag,
    resolve_embeddings_dir,
)
from output_paths import evaluation_output_dir  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_DIR = evaluation_output_dir("first_order")
RESPONSES_XML = ROOT / "eval" / "station_responses.xml"
CHANNEL_NAMES = ["E", "N", "Z"]

# Feature extraction (applied identically to real and synthetic waveforms).
ENV_SMOOTH_SECONDS = 1.0   # moving-average window on the Hilbert envelope
ENV_DECIMATE = 10          # envelope stored at fs/10
SPEC_SMOOTH_BINS = 11      # moving-average window on the linear FAS
SPEC_DECIMATE = 5
LOG_FLOOR = 1e-15

# Response deconvolution (counts -> m/s^2, applied to real AND synthetic).
RESPONSE_OUTPUT = "ACC"
RESPONSE_WATER_LEVEL = 60

# Griffin-Lim defaults (mirror demo/app.py).
GL_MOMENTUM = 0.99
GL_WINDOW = "hann"
GL_RANDOM_STATE = 0

EMBEDDINGS_DIR = resolve_embeddings_dir(None)
_DEFAULT_STFT = {
    "nperseg": 256,
    "noverlap": 192,
    "nfft": 256,
    "resample_hz": 100.0,
    "target_seconds": 70.0,
}
_scale = {}
_source = {"stft": dict(_DEFAULT_STFT)}
_default_scale_path = artifact_path(EMBEDDINGS_DIR, "scale.json")
_default_source_path = artifact_path(EMBEDDINGS_DIR, "source.json")
if _default_scale_path.is_file():
    with _default_scale_path.open(encoding="utf-8") as handle:
        _scale = json.load(handle)
if _default_source_path.is_file():
    with _default_source_path.open(encoding="utf-8") as handle:
        _source = json.load(handle)
STFT_CFG = _source["stft"]
FS = float(STFT_CFG.get("resample_hz", 100.0))
TARGET_SECONDS = float(STFT_CFG.get("target_seconds", 70.0))
TARGET_SAMPLES = int(round(FS * TARGET_SECONDS))
HOP = int(STFT_CFG["nperseg"]) - int(STFT_CFG["noverlap"])
FREQ_BINS = int(STFT_CFG["nfft"]) // 2 + 1

ENV_LEN = int(np.ceil(TARGET_SAMPLES / ENV_DECIMATE))
SPEC_LEN = int(np.ceil((TARGET_SAMPLES // 2 + 1) / SPEC_DECIMATE))


def configure_stft(stft: dict | None) -> None:
    """Set reconstruction geometry from checkpoint-bound embedding provenance."""
    if not stft:
        return
    global STFT_CFG, FS, TARGET_SECONDS, TARGET_SAMPLES, HOP, FREQ_BINS, ENV_LEN, SPEC_LEN
    merged = dict(STFT_CFG)
    merged.update(stft)
    nperseg = int(merged["nperseg"])
    noverlap = int(merged["noverlap"])
    nfft = int(merged["nfft"])
    if nperseg <= 0 or nfft < nperseg or not 0 <= noverlap < nperseg:
        raise ValueError(f"Invalid checkpoint STFT config: {merged}")
    STFT_CFG = merged
    FS = float(merged.get("resample_hz", 100.0))
    TARGET_SECONDS = float(merged.get("target_seconds", 70.0))
    TARGET_SAMPLES = int(round(FS * TARGET_SECONDS))
    HOP = nperseg - noverlap
    FREQ_BINS = nfft // 2 + 1
    ENV_LEN = int(np.ceil(TARGET_SAMPLES / ENV_DECIMATE))
    SPEC_LEN = int(np.ceil((TARGET_SAMPLES // 2 + 1) / SPEC_DECIMATE))


def configure_embeddings_dir(path: str | Path | None) -> Path:
    """Select the embedding export used for metadata and legacy provenance."""
    global EMBEDDINGS_DIR, _scale, _source, STFT_CFG
    EMBEDDINGS_DIR = resolve_embeddings_dir(path)
    scale_path = artifact_path(EMBEDDINGS_DIR, "scale.json")
    if scale_path.is_file():
        with scale_path.open(encoding="utf-8") as handle:
            _scale = json.load(handle)
    else:
        _scale = {}
    with artifact_path(EMBEDDINGS_DIR, "source.json").open(encoding="utf-8") as handle:
        _source = json.load(handle)
    STFT_CFG = dict(_source["stft"])
    configure_stft(STFT_CFG)
    return EMBEDDINGS_DIR


# ── Response deconvolution (runs in worker processes) ─────────────────────────
_INVENTORY = None


def _get_inventory():
    global _INVENTORY
    if _INVENTORY is None:
        from obspy import read_inventory

        _INVENTORY = read_inventory(str(RESPONSES_XML))
    return _INVENTORY


def _find_channel_epoch(station: str, channel_code: str, t):
    """Return (network, location, time) of the response epoch covering t,
    falling back to the most recent epoch with a valid response."""
    fallback = None
    for net in _get_inventory():
        for sta in net:
            if sta.code != station:
                continue
            for cha in sta:
                if cha.code != channel_code or cha.response is None:
                    continue
                if not cha.response.instrument_sensitivity:
                    continue
                start, end = cha.start_date, cha.end_date
                if (t is not None and start is not None and start <= t
                        and (end is None or t <= end)):
                    return net.code, cha.location_code, t
                epoch_time = (start + 86400.0) if start is not None else t
                fallback = (net.code, cha.location_code, epoch_time)
    return fallback


def deconvolve_to_acc(data: np.ndarray, station: str, channel_code: str,
                      event_time_str: str,
                      waveform_domain: str = INSTRUMENT_COUNTS):
    """Convert the recorded waveform domain to ground acceleration in m/s^2."""
    if waveform_domain != INSTRUMENT_COUNTS:
        return physical_acceleration_to_motion(data, "ACC", FS)
    from obspy import Trace, UTCDateTime

    try:
        t = UTCDateTime.strptime(event_time_str, "%Y%m%d%H%M%S")
    except Exception:
        t = None
    epoch = _find_channel_epoch(station, channel_code, t)
    if epoch is None:
        return None
    network, location, starttime = epoch

    trace = Trace(
        data=np.asarray(data, dtype=np.float64),
        header={
            "network": network,
            "station": station,
            "location": location,
            "channel": channel_code,
            "starttime": starttime,
            "sampling_rate": FS,
        },
    )
    try:
        trace.remove_response(
            inventory=_get_inventory(),
            output=RESPONSE_OUTPUT,
            water_level=RESPONSE_WATER_LEVEL,
        )
    except Exception:
        return None
    return trace.data


# ── Feature extraction ────────────────────────────────────────────────────────
def _crop_pad(x: np.ndarray) -> np.ndarray:
    n = TARGET_SAMPLES
    return x[:n] if x.shape[0] >= n else np.pad(x, (0, n - x.shape[0]))


def waveform_features(x: np.ndarray):
    """Log envelope (decimated) and log Fourier amplitude spectrum of one trace."""
    from scipy.signal import hilbert

    x = _crop_pad(np.asarray(x, dtype=np.float64))

    env = np.abs(hilbert(x))
    win = max(1, int(round(ENV_SMOOTH_SECONDS * FS)))
    env = np.convolve(env, np.ones(win) / win, mode="same")
    log_env = np.log(np.maximum(env, LOG_FLOOR))[::ENV_DECIMATE]

    # Fourier amplitude spectrum |X(f)| * dt  (units: m/s^2 per Hz).
    fas = np.abs(np.fft.rfft(x)) / FS
    fas = np.convolve(fas, np.ones(SPEC_SMOOTH_BINS) / SPEC_SMOOTH_BINS, mode="same")
    log_fas = np.log(np.maximum(fas, LOG_FLOOR))[::SPEC_DECIMATE]

    return log_env.astype(np.float32), log_fas.astype(np.float32)


def _process_real(args):
    """Worker: real waveform file -> deconvolved (log_env, log_fas)."""
    path_str, channel, station, channel_code, event_id, waveform_domain = args
    try:
        from obspy import read as obspy_read

        stream = obspy_read(path_str)
        if len(stream) != 3:
            return None
        stream.sort(keys=["channel"])
        trace = stream[channel]
        if abs(trace.stats.sampling_rate - FS) > 1e-6:
            trace.resample(FS)
        data = _crop_pad(trace.data.astype(np.float64))
        acc = deconvolve_to_acc(data, station, channel_code, event_id, waveform_domain)
        if acc is None:
            return None
        return waveform_features(acc)
    except Exception:
        return None


def _process_synth(args):
    """Worker: decoded AE channel -> Griffin-Lim -> features.

    Globally-normalized AEs supply physical magnitude directly, so the legacy
    AmplitudeMLP gain and waveform rescaling are intentionally bypassed.
    """
    decoded, reconstruction, inv_log_gain, amp_scale, metric, gl_iters, station, channel_code, event_id = args
    try:
        import librosa

        m = decoded_to_magnitude(
            np.asarray(decoded, dtype=np.float64), reconstruction,
            legacy_inv_log_gain=float(np.clip(inv_log_gain, 0.1, 20.0)),
        )

        # Match demo/app.py: make sure frame count covers the full window.
        native_frames = int(round(TARGET_SAMPLES / HOP)) + 1
        if native_frames > m.shape[1] > 1:
            x_old = np.linspace(0.0, 1.0, m.shape[1])
            x_new = np.linspace(0.0, 1.0, native_frames)
            m = np.stack([np.interp(x_new, x_old, row) for row in m], axis=0)

        wave = librosa.griffinlim(
            m,
            n_iter=int(gl_iters),
            hop_length=HOP,
            win_length=int(STFT_CFG["nperseg"]),
            n_fft=int(STFT_CFG["nfft"]),
            window=GL_WINDOW,
            center=True,
            momentum=GL_MOMENTUM,
            random_state=GL_RANDOM_STATE,
        ).astype(np.float64)

        wave = postprocess_griffinlim_waveform(
            wave, reconstruction, amp_scale=amp_scale, metric=metric
        )
        wave = _crop_pad(wave)

        acc = deconvolve_to_acc(
            wave, station, channel_code, event_id, reconstruction.waveform_domain
        )
        if acc is None:
            return None
        env, fas = waveform_features(acc)
        return wave.astype(np.float32), env, fas
    except Exception:
        return None


# ── Model loading ─────────────────────────────────────────────────────────────
def find_latest_checkpoint():
    """Most recent diffusion checkpoint dir (by mtime) with a training config."""
    candidates = [
        config.parent
        for config in (DIFF_DIR / "checkpoints").rglob("training_config.json")
        if not any(part.startswith(".") for part in config.parts)
    ]
    if not candidates:
        raise FileNotFoundError("No diffusion checkpoint with training_config.json found.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_diffusion(ckpt_dir: Path):
    from diffusers import DDPMScheduler

    cfg = json.load(open(ckpt_dir / "training_config.json"))
    unet = DiffusionUNet2D.load_pretrained(ckpt_dir).to(DEVICE)
    unet.eval()
    try:
        scheduler = DDPMScheduler.from_pretrained(str(ckpt_dir))
    except Exception:
        scheduler = DDPMScheduler(
            num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02,
            prediction_type="epsilon", clip_sample=False,
        )
    return unet, scheduler, cfg


def load_amplitude():
    ckpt = ROOT / "ML" / "amplitude" / "checkpoints" / "amplitude_mlp.pt"
    stats_path = ROOT / "ML" / "amplitude" / "checkpoints" / "amp_stats.json"
    model = AmplitudeMLP.load(ckpt, device=DEVICE)
    model.eval()
    stats = json.load(open(stats_path))
    return model, stats


# ── Synthetic generation ──────────────────────────────────────────────────────
@torch.no_grad()
def sample_batch(unet, scheduler, cond_batch, data_shape, steps, training_type, seed):
    """Reverse diffusion for a batch. cond_batch: (B, 1, cond_width) on DEVICE."""
    b = cond_batch.shape[0]
    gen = torch.Generator(device=DEVICE).manual_seed(int(seed))
    x = torch.randn(b, *data_shape, device=DEVICE, generator=gen)

    if training_type == "flow_matching":
        num_train = int(getattr(scheduler.config, "num_train_timesteps", 1000))
        num_steps = min(100, steps)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t_cont = 1.0 - i * dt
            t_b = torch.full((b,), round(t_cont * num_train), device=DEVICE, dtype=torch.long)
            x = x - dt * unet.forward(x, t_b, cond_batch).sample
    else:
        scheduler.set_timesteps(steps)
        for t in scheduler.timesteps:
            t_b = torch.full((b,), int(t), device=DEVICE, dtype=torch.long)
            noise_pred = unet.forward(x, t_b, cond_batch).sample
            x = scheduler.step(noise_pred, t, x).prev_sample
    return x


# ── Cache ─────────────────────────────────────────────────────────────────────
def cache_path(ckpt_dir: Path, steps: int, channel: int,
               reconstruction: ReconstructionSpec, ckpt_cfg: dict) -> Path:
    model_tag = diffusion_cache_tag(ckpt_dir, ckpt_cfg, reconstruction)
    tag = (f"{ckpt_dir.parent.name}_{ckpt_dir.name}_steps{steps}_ch{CHANNEL_NAMES[channel]}"
           f"_model{model_tag}_acc")
    return EVAL_DIR / f"cache_{tag}.npz"


def init_or_load_cache(path: Path, test_indices):
    n = len(test_indices)
    if path.exists():
        data = dict(np.load(path))
        if data["indices"].tolist() == list(test_indices):
            return data
        print(f"[eval] {path.name} was built for a different index set; rebuilding.")
    return {
        "indices": np.asarray(test_indices, dtype=np.int64),
        "mags": np.full(n, np.nan, dtype=np.float32),
        "done_real": np.zeros(n, dtype=bool),
        "done_synth": np.zeros(n, dtype=bool),
        "env_real": np.full((n, ENV_LEN), np.nan, dtype=np.float32),
        "spec_real": np.full((n, SPEC_LEN), np.nan, dtype=np.float32),
        "env_synth": np.full((n, ENV_LEN), np.nan, dtype=np.float32),
        "spec_synth": np.full((n, SPEC_LEN), np.nan, dtype=np.float32),
        # Generated waveforms stay in the model's recorded domain. Consumers
        # use waveform_domain to decide whether response removal is needed.
        "wave_synth": np.full((n, TARGET_SAMPLES), np.nan, dtype=np.float32),
    }


def save_cache(path: Path, cache: dict):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + "_tmp.npz")  # np.savez requires .npz suffix
    np.savez(tmp, **cache)  # uncompressed: the waveform block makes compression slow
    tmp.replace(path)


# ── compute stage ─────────────────────────────────────────────────────────────
def run_compute(args):
    metadatas = json.load(open(artifact_path(EMBEDDINGS_DIR, "metadata.json")))
    station_locations = json.load(open(artifact_path(EMBEDDINGS_DIR, "station_locations.json")))
    vs30_path = artifact_path(EMBEDDINGS_DIR, "station_vs30.json")
    station_vs30 = json.load(open(vs30_path)) if vs30_path.exists() else None

    channel = CHANNEL_NAMES.index(args.channel)

    def _rel(p):
        try:
            return str(Path(p).resolve().relative_to(ROOT))
        except ValueError:
            return str(p)

    ckpt_dir = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint()
    unet, scheduler, ckpt_cfg = load_diffusion(ckpt_dir)
    checkpoint_scale = ckpt_cfg.get("embedding_scale", _scale)
    checkpoint_split = ckpt_cfg.get("split", checkpoint_scale)
    test_indices_value = checkpoint_split.get("val_indices")
    if test_indices_value is None:
        raise ValueError(
            "Diffusion checkpoint does not record validation indices and no legacy "
            "scale.json with val_indices was found."
        )
    all_test_indices = list(test_indices_value)
    test_indices = deterministic_fraction_subset(
        all_test_indices, fraction=args.fraction, limit=args.limit
    )
    print(
        f"[eval] deterministic validation sample: {len(test_indices)}/"
        f"{len(all_test_indices)} records (fraction={args.fraction:g})"
    )
    training_type = ckpt_cfg.get("training_type", "ddpm")
    data_shape = tuple(ckpt_cfg["data_shape"])
    data_mode = ckpt_cfg.get("data_mode", "latent")
    data_normalization = ckpt_cfg.get("data_normalization", {})
    emb_std = float(data_normalization.get("std", ckpt_cfg.get("emb_std", 1.0)))
    emb_mean = float(data_normalization.get("mean", ckpt_cfg.get("emb_mean", 0.0)))
    ckpt_source = "--checkpoint" if args.checkpoint else "default: latest checkpoint by mtime"
    print(f"[eval] diffusion model: {_rel(ckpt_dir)}  ({ckpt_source})")
    print(f"[eval]   type={training_type}  mode={data_mode}  shape={data_shape}  "
          f"steps={args.steps}")

    source_path = ckpt_dir / "embedding_source.json"
    if not source_path.exists():
        source_path = artifact_path(EMBEDDINGS_DIR, "source.json")
    reconstruction = resolve_reconstruction_spec(
        ckpt_cfg, embeddings_source_path=source_path,
        ae_checkpoint_override=args.ae_checkpoint,
        waveform_domain_override=(
            None if args.waveform_domain == "auto" else args.waveform_domain
        ),
    )
    print(f"[eval] AE normalization: {reconstruction.mode} "
          f"(source={reconstruction.source_origin}, id={reconstruction.source_identity})")
    print(f"[eval] waveform domain: {reconstruction.waveform_domain}")
    if reconstruction.waveform_domain == INSTRUMENT_COUNTS and not RESPONSES_XML.exists():
        raise FileNotFoundError(
            f"Missing {RESPONSES_XML}. Run eval/fetch_station_responses.py first."
        )
    configure_stft(checkpoint_stft_config(ckpt_cfg, embeddings_source_path=source_path))
    print(f"[eval] STFT reconstruction: nperseg={STFT_CFG['nperseg']} "
          f"noverlap={STFT_CFG['noverlap']} nfft={STFT_CFG['nfft']} "
          f"fs={FS:g}Hz duration={TARGET_SECONDS:g}s")

    ae_model = None
    if data_mode == "latent":
        ae_ckpt = reconstruction.ae_checkpoint
        if not ae_ckpt:
            raise ValueError("Latent diffusion evaluation needs an AE checkpoint in checkpoint provenance "
                             "or --ae_checkpoint.")
        ae_source = "--ae_checkpoint" if args.ae_checkpoint else reconstruction.source_origin
        print(f"[eval] AE decoder: {_rel(ae_ckpt)}  ({ae_source})")
        ae_model, _ = load_model(ae_ckpt, device=DEVICE)
        ae_model.eval()

    amp_model = amp_stats = None
    amp_metric = "max"
    if reconstruction.uses_amplitude_model:
        amp_model, amp_stats = load_amplitude()
        amp_metric = amp_stats.get("metric", "std")
        log_std_mean = torch.tensor(amp_stats["log_std_mean"], dtype=torch.float32)
        log_std_scale = torch.tensor(amp_stats["log_std_scale"], dtype=torch.float32)
        gain_mean = torch.tensor(amp_stats["gain_mean"], dtype=torch.float32)
        gain_scale = torch.tensor(amp_stats["gain_scale"], dtype=torch.float32)

    condition_normalization = ckpt_cfg.get("conditioning_normalization", {})
    cond_mean_value = condition_normalization.get("mean", checkpoint_scale.get("cond_mean"))
    cond_std_value = condition_normalization.get("std", checkpoint_scale.get("cond_std"))
    if cond_mean_value is None or cond_std_value is None:
        raise ValueError(
            "Diffusion checkpoint does not record conditioning normalization and no "
            "compatible legacy scale.json was found."
        )
    cond_mean = torch.tensor(cond_mean_value, dtype=torch.float32)
    cond_std = torch.tensor(cond_std_value, dtype=torch.float32).clamp(min=1e-8)

    diff_nc = int(unet.num_continuous)
    amp_nc = int(amp_model.num_continuous) if amp_model is not None else 0
    need_vs30 = max(diff_nc, amp_nc) >= 7
    if need_vs30 and station_vs30 is None:
        raise FileNotFoundError("Checkpoint expects Vs30 conditioning but "
                                "embeddings/station_vs30.json is missing.")

    def build_conds(meta):
        """Raw conditioning -> (diffusion cond row, legacy amplitude row)."""
        full = create_conditioning_vector(
            meta, station_locations, station_vs30 if need_vs30 else None
        )
        nc_full = full.shape[0] - 2  # continuous block excl. station/channel idx

        def normed(nc):
            v = full.clone()
            v[:nc] = (v[:nc] - cond_mean[:nc]) / cond_std[:nc]
            return v

        d = normed(diff_nc)
        parts = [d[:diff_nc], d[nc_full:nc_full + 1]]          # continuous + station
        if unet.use_channel:
            parts.append(d[nc_full + 1:nc_full + 2])           # channel idx
        diff_cond = torch.cat(parts)
        if amp_model is None:
            return diff_cond, None
        a = normed(amp_nc)
        amp_cond = torch.cat([a[:amp_nc], a[nc_full:nc_full + 1]])
        return diff_cond, amp_cond

    path = cache_path(ckpt_dir, args.steps, channel, reconstruction, ckpt_cfg)
    selection_digest = deterministic_selection_tag(test_indices)
    path = path.with_name(f"{path.stem}_n{len(test_indices)}_sel{selection_digest}.npz")
    cache = init_or_load_cache(path, test_indices)
    cache["waveform_domain"] = np.asarray(reconstruction.waveform_domain)
    for j, idx in enumerate(test_indices):
        cache["mags"][j] = float(metadatas[idx]["magnitude"])

    def resolve(m):
        p = Path(m["file_path"])
        return str(p if p.is_absolute() else (DIFF_DIR / p).resolve())

    def response_args(meta):
        channel_code = f"{meta.get('channel_type', 'HH')}{CHANNEL_NAMES[channel]}"
        return meta["station_name"], channel_code, str(meta.get("event_id", ""))

    # ── Real pass ────────────────────────────────────────────────────────────
    todo_real = [j for j in range(len(test_indices)) if not cache["done_real"][j]]
    if todo_real:
        print(f"[eval] real pass: {len(todo_real)} waveforms ({args.num_workers} workers)")
        jobs = [
            (resolve(metadatas[test_indices[j]]), channel,
             *response_args(metadatas[test_indices[j]]), reconstruction.waveform_domain)
            for j in todo_real
        ]
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            for j, out in zip(todo_real, tqdm(
                    pool.map(_process_real, jobs, chunksize=16), total=len(jobs))):
                if out is not None:
                    cache["env_real"][j], cache["spec_real"][j] = out
                cache["done_real"][j] = True
        n_failed = int(np.isnan(cache["env_real"][todo_real]).all(axis=1).sum())
        if n_failed:
            print(f"[eval] real pass: {n_failed} waveforms failed (read/response)")
        save_cache(path, cache)

    # ── Synthetic pass ───────────────────────────────────────────────────────
    todo_synth = [j for j in range(len(test_indices)) if not cache["done_synth"][j]]
    print(f"[eval] synthetic pass: {len(todo_synth)} samples, batch={args.batch_size}")
    t0 = time.time()
    n_done = 0
    batches_since_save = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        for start in range(0, len(todo_synth), args.batch_size):
            rows = todo_synth[start:start + args.batch_size]
            metas = [metadatas[test_indices[j]] for j in rows]
            conds = [build_conds(m) for m in metas]
            diff_cond = torch.stack([c[0] for c in conds]).unsqueeze(1).to(DEVICE)
            if amp_model is not None:
                amp_cond = torch.stack([c[1] for c in conds]).to(DEVICE)
                with torch.no_grad():
                    raw_pred = amp_model(amp_cond).cpu()
                amp_scales = torch.exp(raw_pred[:, :3] * log_std_scale + log_std_mean)
                if raw_pred.shape[1] >= 6:
                    gains = (raw_pred[:, 3:6] * gain_scale + gain_mean).clamp(0.1, 20.0)
                else:
                    gains = torch.full_like(amp_scales, 1.0)
            else:
                amp_scales = gains = None

            x = sample_batch(unet, scheduler, diff_cond, data_shape,
                             args.steps, training_type, seed=args.seed + start)
            x = x * emb_std + emb_mean
            if data_mode == "latent":
                with torch.no_grad():
                    specs = ae_model.decode(x)
            else:
                specs = x
            # Do not clip the decoded normalized output: values above one are
            # valid extrapolated global log-amplitudes.
            specs = specs[:, :3, :FREQ_BINS, :].cpu().numpy()

            jobs = [
                (specs[k, channel], reconstruction,
                 float(gains[k, channel]) if gains is not None else 1.0,
                 float(amp_scales[k, channel]) if amp_scales is not None else None,
                 amp_metric, args.gl_iters,
                 *response_args(metas[k]))
                for k in range(len(rows))
            ]
            for j, out in zip(rows, pool.map(_process_synth, jobs)):
                if out is not None:
                    cache["wave_synth"][j], cache["env_synth"][j], cache["spec_synth"][j] = out
                cache["done_synth"][j] = True

            n_done += len(rows)
            batches_since_save += 1
            if batches_since_save >= args.save_every or n_done >= len(todo_synth):
                save_cache(path, cache)
                batches_since_save = 0
            rate = n_done / max(time.time() - t0, 1e-9)
            eta_min = (len(todo_synth) - n_done) / max(rate, 1e-9) / 60.0
            print(f"[eval] synth {n_done}/{len(todo_synth)}  "
                  f"({rate:.2f} samples/s, ETA {eta_min:.0f} min)")

    print(f"[eval] cache written: {path}")
    return path


# ── plot stage ────────────────────────────────────────────────────────────────
# Validated categorical palette (dataviz reference, slots 1-8, fixed order).
BIN_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
              "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
MIN_BIN_COUNT = 5


def _smooth_curve(y: np.ndarray, win: int) -> np.ndarray:
    """Edge-preserving moving average (odd window) for display smoothing."""
    if win <= 1:
        return y
    win = win + 1 - (win % 2)
    pad = win // 2
    yp = np.pad(y, pad, mode="edge")
    return np.convolve(yp, np.ones(win) / win, mode="valid")


def run_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.cache:
        path = Path(args.cache)
    else:
        caches = sorted(EVAL_DIR.glob("cache_*.npz"), key=lambda p: p.stat().st_mtime)
        if not caches:
            raise FileNotFoundError(f"No cache in {EVAL_DIR}; run `compute` first.")
        path = caches[-1]
    with np.load(path) as data:
        cache = {k: data[k] for k in data.files if k != "wave_synth"}
    print(f"[eval] plotting from {path.name}  "
          f"(real {int(cache['done_real'].sum())}, synth {int(cache['done_synth'].sum())})")

    edges = [float(v) for v in args.bins.split(",")]
    if len(edges) - 1 > len(BIN_COLORS):
        raise ValueError(f"At most {len(BIN_COLORS)} bins supported; got {len(edges) - 1}.")
    mags = cache["mags"]

    # Crop the display ranges: the last seconds / highest frequencies carry
    # crop-taper and Griffin-Lim band-edge artefacts, not signal.
    t_axis = np.arange(ENV_LEN) * ENV_DECIMATE / FS
    df = FS / TARGET_SAMPLES
    f_axis = np.arange(SPEC_LEN) * SPEC_DECIMATE * df
    t_mask = t_axis <= args.max_time if args.max_time > 0 else np.ones_like(t_axis, bool)
    f_mask = f_axis <= args.max_freq if args.max_freq > 0 else np.ones_like(f_axis, bool)
    t_axis, f_axis = t_axis[t_mask], f_axis[f_mask]

    channel = path.stem.rsplit("_ch", 1)[-1].split("_")[0]
    panels = [
        ("a)", "Real Data", "env_real", t_axis, "Time [s]"),
        ("b)", "Diffusion Synthetics", "env_synth", t_axis, "Time [s]"),
        ("c)", "Real Data", "spec_real", f_axis, "Frequency [Hz]"),
        ("d)", "Diffusion Synthetics", "spec_synth", f_axis, "Frequency [Hz]"),
    ]
    ylabels = {
        "env": r"Log-Amplitude [$m/s^2$]",
        "spec": r"Log-Amplitude [$m/s^2\ Hz^{-1}$]",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex="row", sharey="row")
    fig.patch.set_facecolor("white")

    legend_handles, legend_labels = [], []
    for ax, (tag, title, key, x_axis, xlabel) in zip(axes.ravel(), panels):
        data = cache[key][:, t_mask if key.startswith("env") else f_mask]
        for b in range(len(edges) - 1):
            lo, hi = edges[b], edges[b + 1]
            sel = (mags >= lo) & (mags < hi) & ~np.isnan(data).all(axis=1)
            n = int(sel.sum())
            if n < MIN_BIN_COUNT:
                continue
            mu = np.nanmean(data[sel], axis=0)
            sd = np.nanstd(data[sel], axis=0)
            if key.startswith("spec") and args.spec_smooth_hz > 0:
                # Same smoothing on both panels: Griffin-Lim phase artefacts
                # share a random seed across samples, so they survive the
                # ensemble mean and need display smoothing to read the trend.
                win = int(round(args.spec_smooth_hz / (SPEC_DECIMATE * df)))
                mu, sd = _smooth_curve(mu, win), _smooth_curve(sd, win)
            line, = ax.plot(x_axis, mu, color=BIN_COLORS[b], lw=1.6)
            ax.fill_between(x_axis, mu - sd, mu + sd, color=BIN_COLORS[b],
                            alpha=0.16, lw=0)
            label = f"{lo:g}–{hi:g}"
            if key == "env_real" and label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(f"{label} (n={n})")
        ax.set_title(title, color=INK, fontsize=12)
        ax.text(-0.1, 1.06, tag, transform=ax.transAxes, fontsize=12, color=INK)
        ax.set_xlabel(xlabel, color=INK)
        ax.set_ylabel(ylabels["env" if key.startswith("env") else "spec"], color=INK)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(MUTED)

    fig.legend(legend_handles, legend_labels, title="Magnitude bins",
               loc="lower center", ncol=min(6, len(legend_labels)),
               frameon=True, edgecolor=GRID)
    fig.suptitle(
        f"First-order seismogram characteristics — test split, "
        f"{channel} component", color=INK, fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])

    out = EVAL_DIR / f"first_order_{path.stem.removeprefix('cache_')}.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"[eval] figure saved: {out}")
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("stage", choices=["compute", "plot", "all"], nargs="?",
                        default="all")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Diffusion checkpoint dir (default: most recent).")
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=str(EMBEDDINGS_DIR),
        help="Embedding export directory used for metadata and station artifacts.",
    )
    parser.add_argument("--ae_checkpoint", type=str, default=None,
                        help="Autoencoder checkpoint used to decode latents. "
                             "Default: the one recorded by the diffusion checkpoint "
                             "(legacy fallback: embeddings/source.json). "
                             "Must be the AE the diffusion latents were created with.")
    parser.add_argument(
        "--waveform_domain",
        choices=["auto", "instrument_counts", "physical_acceleration"],
        default="auto",
        help="Override checkpoint waveform-domain provenance for legacy checkpoints.",
    )
    parser.add_argument("--steps", type=int, default=1000,
                        help="DDPM inference steps (flow matching caps at 100).")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0,
                        help="Evaluate only N evenly-spaced test samples (0 = all).")
    parser.add_argument(
        "--fraction", type=float, default=1.0,
        help="Deterministic evenly-spaced fraction of validation records in (0, 1].",
    )
    parser.add_argument("--channel", choices=CHANNEL_NAMES, default="E",
                        help="Component to evaluate (paper uses East-West).")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gl_iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=4,
                        help="Write the cache every N batches during the synthetic pass.")
    parser.add_argument("--bins", type=str, default="1,1.5,2,2.5,3,3.5,4.5,6",
                        help="Comma-separated magnitude bin edges (max 8 bins).")
    parser.add_argument("--spec_smooth_hz", type=float, default=3.0,
                        help="Display smoothing window (Hz) for the spectrum "
                             "panels, applied to real and synthetic alike "
                             "(<=0 disables).")
    parser.add_argument("--max_time", type=float, default=60.0,
                        help="Envelope panels show 0..max_time seconds (<=0 = full window).")
    parser.add_argument("--max_freq", type=float, default=40.0,
                        help="Spectrum panels show 0..max_freq Hz (<=0 = Nyquist).")
    parser.add_argument("--cache", type=str, default=None,
                        help="Cache .npz for `plot` (default: most recent).")
    args = parser.parse_args()
    configure_embeddings_dir(args.embeddings_dir)

    if args.stage in ("compute", "all"):
        path = run_compute(args)
        if args.stage == "all" and args.cache is None:
            args.cache = str(path)
    if args.stage in ("plot", "all"):
        run_plot(args)


if __name__ == "__main__":
    main()
