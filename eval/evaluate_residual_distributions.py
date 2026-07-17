"""
Residual distributions of peak amplitudes (GWM paper, Fig. 5).

Following Section 4.3.2 of the paper: fit a simple reference GMM to the real
data by ordinary least squares,

    log10(Y) = a0 + a1*M + a2*log10(Vs30) + a3*log10(R_hyp)     (eqs. 10/11)

for Y in {PGA, PGV}, then subtract this reference from (i) the real data,
(ii) the diffusion synthetics, and (iii) a published GMM's median predictions,
all on the identical record set. The figure overlays the three residual
distributions (histogram + KDE) with box plots underneath; the headline
comparison is the standard deviation of each distribution: if the GWM has
learned the aleatory variability of the data, sigma(GWM) ~ sigma(real),
whereas a median GMM is narrower by construction.

Everything is read from the cache written by evaluate_peak_amplitudes.py
`compute` (run that first, with the same --gmm); no waveforms are touched.
The record set is the matched intersection: test split, M >= the GMM's
validity floor, plausible Vs30, all measures finite.

Notes:
  - --obs picks the observed/real measure: 'e' (E component, identical
    definition to the synthetics; default) or 'rot' (RotD50, identical
    definition to the GMM predictions). The GMM population always uses the
    model's own horizontal definition, so with --obs e its residual mean
    carries a small component-definition offset; spreads are unaffected.
  - The reference model is refit on whichever measure --obs selects.

Run from the project root:
    python eval/evaluate_residual_distributions.py [--gmm edwardsfah13]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_peak_amplitudes import (  # noqa: E402
    GMM_REGISTRY, OUT_DIR, VS30_RANGE,
    GWM_COLOR, GMM_COLOR, INK, MUTED, GRID,
)

REAL_COLOR = "#63615c"


def fit_reference(mag, vs30, r_hyp, log_y):
    """OLS fit of log10(Y) = a0 + a1*M + a2*log10(Vs30) + a3*log10(R)."""
    X = np.column_stack([np.ones_like(mag), mag, np.log10(vs30), np.log10(r_hyp)])
    coef, *_ = np.linalg.lstsq(X, log_y, rcond=None)
    return coef, X @ coef


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--gmm", choices=sorted(GMM_REGISTRY), default="edwardsfah13",
                        help="GMM population to include (its predictions must "
                             "already be in the peaks cache).")
    parser.add_argument("--gmm_min_mag", type=float, default=None,
                        help="Magnitude floor (default: the model's validity floor).")
    parser.add_argument("--obs", choices=["e", "rot"], default="e",
                        help="Observed measure: E component (matches synthetics) "
                             "or RotD50 (matches GMM definitions).")
    parser.add_argument("--peaks_cache", type=str, default=None,
                        help="Peaks .npz (default: most recent in eval/peak_amplitudes).")
    parser.add_argument("--xlim", type=float, default=3.0)
    parser.add_argument("--bin_width", type=float, default=0.1,
                        help="Histogram bin width in log10 units.")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    if args.peaks_cache:
        path = Path(args.peaks_cache)
    else:
        caches = sorted(OUT_DIR.glob("peaks_*.npz"), key=lambda p: p.stat().st_mtime)
        if not caches:
            raise FileNotFoundError(f"No cache in {OUT_DIR}; run "
                                    "evaluate_peak_amplitudes.py compute first.")
        path = caches[-1]
    cache = dict(np.load(path))

    gmm = GMM_REGISTRY[args.gmm]
    min_mag = args.gmm_min_mag if args.gmm_min_mag is not None else gmm["min_mag"]
    pga_gmm_key, pgv_gmm_key = f"pga_gmm_{args.gmm}", f"pgv_gmm_{args.gmm}"
    if pga_gmm_key not in cache:
        raise KeyError(f"No {args.gmm} predictions in {path.name}; run "
                       f"evaluate_peak_amplitudes.py compute --gmm {args.gmm} first.")

    obs_suffix = "e" if args.obs == "e" else "rot"
    keys = {
        "PGA": (f"pga_obs_{obs_suffix}", "pga_synth_e", pga_gmm_key),
        "PGV": (f"pgv_obs_{obs_suffix}", "pgv_synth_e", pgv_gmm_key),
    }

    # Matched record set: every population defined on every record.
    sel = (cache["in_val"].astype(bool) & (cache["mag"] >= min_mag)
           & (cache["vs30"] >= VS30_RANGE[0]) & (cache["vs30"] <= VS30_RANGE[1]))
    for im_keys in keys.values():
        for key in im_keys:
            sel &= np.isfinite(cache[key]) & (cache[key] > 0)
    n = int(sel.sum())
    if n < 10:
        raise RuntimeError(f"Only {n} matched records in {path.name}.")
    print(f"[eval] {path.name}: {n} matched records "
          f"(test split, M>={min_mag:g}, obs measure: {args.obs})")

    mag, vs30, r_hyp = cache["mag"][sel], cache["vs30"][sel], cache["r_hyp"][sel]
    populations = [
        ("Real data", REAL_COLOR),
        ("GWM synthetics", GWM_COLOR),
        (gmm["label"], GMM_COLOR),
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(11, 6.5), sharex="col",
        gridspec_kw={"height_ratios": [3.0, 1.1]},
    )
    fig.patch.set_facecolor("white")

    for col, (im, (obs_key, synth_key, gmm_key)) in enumerate(keys.items()):
        coef, ref = fit_reference(mag, vs30, r_hyp,
                                  np.log10(cache[obs_key][sel]))
        print(f"[eval] fitted reference: log10({im}) = {coef[0]:+.4f} "
              f"{coef[1]:+.4f}*M {coef[2]:+.4f}*log10(Vs30) "
              f"{coef[3]:+.4f}*log10(R)")
        residuals = [np.log10(cache[key][sel]) - ref
                     for key in (obs_key, synth_key, gmm_key)]
        for (name, _), res in zip(populations, residuals):
            print(f"[eval]   {im} {name}: sigma={res.std():.3f}  "
                  f"median={np.median(res):+.3f}")

        ax = axes[0, col]
        bins = np.arange(-args.xlim, args.xlim + args.bin_width, args.bin_width)
        grid = np.linspace(-args.xlim, args.xlim, 400)
        for (name, color), res in zip(populations, residuals):
            ax.hist(res, bins=bins, density=True, histtype="stepfilled",
                    color=color, alpha=0.22, lw=0)
            ax.plot(grid, gaussian_kde(res)(grid), color=color, lw=1.6,
                    label=f"{name} ($\\sigma$={res.std():.2f})")
        ax.text(-0.06, 1.04, "ab"[col] + ")", transform=ax.transAxes,
                fontsize=12, color=INK)
        ax.set_title(im, color=INK, fontsize=12)
        ax.set_ylabel("Density", color=INK)
        ax.legend(fontsize=8, frameon=True, edgecolor=GRID)

        ax = axes[1, col]
        box = ax.boxplot(
            residuals[::-1], vert=False, positions=[1, 2, 3], widths=0.55,
            patch_artist=True,
            flierprops=dict(marker="x", markersize=3, markeredgecolor=MUTED),
            medianprops=dict(color=INK, lw=1.2),
            whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
        )
        for patch, (_, color) in zip(box["boxes"], populations[::-1]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor(INK)
        ax.set_yticks([3, 2, 1])
        ax.set_yticklabels([p[0] for p in populations], fontsize=8, color=INK)
        ax.axvline(0.0, color=GRID, lw=0.8, zorder=0)
        ax.set_xlim(-args.xlim, args.xlim)
        ax.set_xlabel(r"Residuals (Log$_{10}$ scale)", color=INK)

        for ax in axes[:, col]:
            ax.grid(True, color=GRID, lw=0.6)
            ax.tick_params(colors=MUTED)
            for spine in ax.spines.values():
                spine.set_color(MUTED)

    obs_name = "E component" if args.obs == "e" else "RotD50"
    fig.suptitle(
        f"Peak amplitude residuals vs OLS reference — test split, "
        f"M$\\geq${min_mag:g}, {obs_name} (n={n:,})",
        color=INK, fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = OUT_DIR / (f"fig_residuals_{path.stem.removeprefix('peaks_')}"
                     f"_{args.gmm}_{args.obs}.png")
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"[eval] figure saved: {out}")


if __name__ == "__main__":
    main()
