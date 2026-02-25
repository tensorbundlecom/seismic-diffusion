# Terminology Correction: `W` Usage in Existing Experiments

Date: 2026-02-24

## Scope

This note corrects historical `W*` naming and defines current naming policy.

## Correction

Current legacy-condition folders:
- `LegacyCondBaseline` (formerly `WBaseline`)
- `LegacyCondFullCov` (formerly `WFullCov`)
- `LegacyCondFlow` (formerly `NormalizingW`)
- `LegacyCondAblation` (formerly `WAblation`)

`W` is **not** a StyleGAN latent `W-space`.

It is a **deterministic condition embedding**:
- built from physical condition + station embedding
- used as a conditioning feature in encoder/flow/decoder paths

So these models are condition-embedding variants, not true `W-space latent` models.

## Why it matters

A true StyleGAN-like `W-space` is a latent variable space sampled via a stochastic base latent (e.g., `u ~ N(0,I)`) then mapped to `w`.
In existing `W*` experiments, `w` does not play that role.

## Naming policy (from now on)

- Use `cond_embed` for deterministic conditioning vectors.
- Reserve `W-space` naming for models where `w` is generated from stochastic latent and acts as primary latent variable.

## Folder naming outcome

Historical `W*` folder names were renamed to `LegacyCond*` to avoid semantic confusion with true StyleGAN-like W-space.
True stochastic W-latent work now lives in:
- `ML/autoencoder/experiments/WSpaceVAE/`
