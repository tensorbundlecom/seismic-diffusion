# Physical-unit waveform preprocessing

**Last updated:** 2026-08-25

## TL;DR

This workflow creates a new MiniSEED archive in physical acceleration units (`m/s^2`) from raw digitizer counts before training a globally normalized autoencoder. It selects the same default corpus as the current AE dataset—maximum component SNR strictly greater than 2 and no component gaps (currently 108,421 files)—but does not modify or mix with legacy count-domain archives. A strict, epoch-aware StationXML coverage gate must succeed before correction; QC then emits CSV/JSON summaries and deterministic raw-versus-corrected PNG reviews.

## Why this is a separate dataset

Global spectral normalization preserves amplitude differences. Training it on count-domain traces would therefore encode instrument gain and response differences along with ground motion. Response removal puts all accepted traces into a common physical unit, but it changes the signal and requires a new AE, new embeddings, and new diffusion training.

Never combine response-corrected outputs with raw or previously filtered count-domain waveforms. The scripts deliberately create a new archive and refuse untracked or configuration-mismatched output directories.

## Data contract and response-coverage gate

The input is raw MiniSEED under `data/waveforms`; it is not the legacy filtered archive. By default, selection is from `data/waveform_summary.csv`:

- `max(snr1, snr2, snr3) > 2.0` (strictly greater than 2);
- `max(gap1, gap2, gap3) == 0`;
- the current selection contains 108,421 waveform files.

For every trace in every selected three-component file, [fetch_station_responses.py](../../eval/fetch_station_responses.py) reads the raw header and requires one StationXML response epoch matching its full NSLC identity:

```text
network.station.location.channel
```

The epoch must cover the trace recording interval. A matching station, channel family, or response from a different location code or epoch does not count. Unreadable files and files that are not exactly three component are retained as incomplete in the coverage denominator. `--strict` exits nonzero unless every selected file and every component is covered, so it is the required gate before physical correction.

### Blank network headers

Some legacy raw MiniSEED traces have a blank network header. The default is an explicit, provenance-preserving fallback: `--default-network KO` uses `KO` only for StationXML fetch/match and response removal; it does not modify the raw archive. Reports preserve both the raw/header NSLC and the effective NSLC, and the converter manifest/results record the fallback and inferred-trace count.

This fallback is supported by the current corpus: all 325,212 populated raw network headers are `KO`, and exact `KO` response epochs cover all 51 blank-header traces in 17 files. It is still an explicit assumption, not an identifier rewrite. To reject rather than infer blank network codes, run the inventory command with `--default-network ""` (an empty string) and run the converter with `--default-network none`; the affected traces will remain uncovered or fail correction. The wrapper passes `--default-network KO` explicitly.

The inventory is merged and resumable. Its default artifacts live beside `eval/station_responses.xml`:

- `station_response_manifest.json` — exact trace request inventory, including raw/effective NSLC and blank-network fallback provenance, and the selected-file fingerprint;
- `station_response_coverage.csv` and `station_response_coverage.json` — per-trace/per-file coverage and uncovered NSLC counts;
- `station_response_fetch_cache.json` — cached FDSN request results;
- `station_responses.xml` — merged StationXML response inventory.

The currently available local inventory may be incomplete, especially for HN, EH, and BH. A full inventory for those channel families requires successful access to an FDSN service providing the relevant network metadata. Do not bypass the strict gate or substitute a response from another channel/epoch.

## Processing contract

For each accepted trace, [remove_instrument_response.py](../../preprocessing/remove_instrument_response.py) applies this exact order:

1. Demean.
2. Linear detrend.
3. Apply a 5% cosine taper.
4. Select the exact full-ID/time response from the StationXML inventory and remove it to `ACC` (`m/s^2`).
5. Apply the common 2–15 Hz bandpass forwards and backwards (four corners, zero phase by default).

Default response-removal parameters are:

| Parameter | Default |
| --- | --- |
| Output unit | `ACC` / `m/s^2` |
| Response pre-filter | `0.5, 1.0, 20.0, 22.5` Hz |
| Water level | `None`; stabilize with the explicit pre-filter |
| Taper | cosine, `0.05` |
| Final bandpass | `2.0`–`15.0` Hz |
| Bandpass corners | `4` |
| Zero-phase bandpass | enabled |

The default avoids a deconvolution water level because this mixed-instrument
archive requests acceleration from sensors whose native quantity may instead
be velocity. ObsPy warns that a water level can suppress valid spectrum in
that situation; `--water-level` remains available for controlled experiments.

A file is rejected as a whole if it is not exactly three-component, a component lacks a valid response, its final filter would violate Nyquist, or processing creates non-finite samples. Corrected MiniSEED is written as floating-point (`FLOAT64`) so physical values are not quantized back into the raw encoding.

## Full workflow

Run the wrapper from the project root:

```bash
bash source/prepare_physical_waveforms.sh
```

It performs the following equivalent commands. `PYTHON_BIN`, `WORKERS`, `OUTPUT_DIR`, `QC_SAMPLE_COUNT`, and `QC_SEED` may be overridden as environment variables; defaults are `.venv/bin/python`, `4`, `data/physical_waveforms_snr2_2-15hz`, `6`, and `20260822`.

```bash
# 1. Populate/validate full NSLC-and-epoch coverage. This must pass.
.venv/bin/python eval/fetch_station_responses.py --default-network KO --strict

# 2. Create a separate acceleration archive; raw data are never changed in place.
.venv/bin/python preprocessing/remove_instrument_response.py \
  --output-dir data/physical_waveforms_snr2_2-15hz \
  --default-network KO \
  --workers 4

# 3. Run the archive QC and render a deterministic sample of six review PNGs.
MPLCONFIGDIR=/tmp/seismic-mpl-cache \
.venv/bin/python preprocessing/qc_physical_waveforms.py \
  --corrected-dir data/physical_waveforms_snr2_2-15hz \
  --raw-dir data/waveforms \
  --manifest data/physical_waveforms_snr2_2-15hz/response_removal_manifest.json \
  --output-dir data/physical_waveforms_snr2_2-15hz/qc \
  --sample-count 6 \
  --seed 20260822
```

For a small smoke test, use `--limit-files N` for the response inventory and
`--limit N` for response removal, and choose a separate output directory. The
full-run response-removal manifest records the full selected-file count, so a
limited run cannot safely resume into a full-run archive.

## Output archive, provenance, and resuming

The default physical archive is `data/physical_waveforms_snr2_2-15hz`. Its MiniSEED layout mirrors `data/waveforms`, while run-level artifacts are:

- `response_removal_manifest.json` — immutable configuration, SHA-256 values for the summary and StationXML inventory, selected-file count, blank-network fallback, processing parameters, and run history;
- `response_removal_results.csv` — one row per attempted file: raw path, output path, `processed`, `skipped_existing`, or `failed`, a reason on failure, and its inferred-network trace count.

Writes are atomic. A later run skips an existing output only if the directory has a manifest whose configuration hash and selected-file count exactly match the requested run. A non-empty output directory without that manifest, a changed configuration, or a changed selection count is rejected. Use a fresh `--output-dir` for any changed preprocessing parameters. `--overwrite` reprocesses existing files only under the same manifest configuration.

Before proceeding to AE training, inspect `response_removal_results.csv`; any `failed` rows make the removal command return a nonzero status.

## Quality control and visual review

[qc_physical_waveforms.py](../../preprocessing/qc_physical_waveforms.py) is read-only: it audits the corrected archive and writes a reproducible QC record. Supplying the response-removal manifest is strongly recommended, because it carries the expected `m/s^2` unit and the raw-to-physical provenance.

The default thresholds flag pathological deconvolutions rather than ordinary strong motion:

| Check | Default failure condition |
| --- | --- |
| Trace structure | Not exactly 3 traces |
| Values | Any non-finite sample |
| Unit/provenance | Missing or not `m/s^2` |
| Gaps | Any gap (`--max-gap-seconds 0`) |
| Extreme amplitude | max absolute amplitude > `100 m/s^2` |
| Narrow spike | peak/RMS > `100` |
| Possible clipping | contiguous extremum plateau ≥ 5 samples and plateau fraction ≥ `1e-3` |

QC writes:

- `qc/qc_summary.csv` — one row per corrected waveform and all metrics/issues;
- `qc/qc_summary.json` — thresholds, counts, issue counts, seed, and selected review paths;
- `qc/review_png/*.png` — deterministic randomly selected successful records.

The default six PNGs are chosen with a seeded PRNG (`20260822`). Each plot shows each component’s raw counts and response-corrected acceleration in both time and frequency domains, with the final 2–15 Hz passband shaded. Inspect these images and the failure rows before training. Generated preview data and images are Git-ignored; they are run artifacts, not source data.

## Next step

Only after strict response coverage, zero response-removal failures, and acceptable QC should this archive replace the count-domain input to a new globally normalized AE run. Physical acceleration is normally far below one in SI units, so use the epsilon-aware log transform rather than legacy `log1p`:

```bash
cd ML/autoencoder
python train.py \
  --data_dir ../../data/physical_waveforms_snr2_2-15hz \
  --waveform_domain physical_acceleration \
  --global_normalization \
  --name vae-global-physical-v2
```

With `physical_acceleration`, an omitted `--amplitude_epsilon` resolves to `1e-12`; pass an explicit positive value only to intentionally override that default. If this AE uses `--use_phasenet_perceptual`, global normalization is mandatory because PhaseNet receives the exact inverse physical magnitude reconstructed from these fitted bounds. The epsilon, fitted bounds, archive manifest, embeddings, and diffusion checkpoint form one provenance chain and must remain associated.

## Related docs

- [Documentation index](../README.md)
- [Project overview](../../README.md)
- [Glossary](../glossary.md)
- [Changelog](../changelog.md)
