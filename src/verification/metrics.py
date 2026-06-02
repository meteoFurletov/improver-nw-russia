"""Forecast verification metrics.

Implements the standard WMO-recommended verification metrics for
probabilistic and deterministic forecast evaluation.

Primary metric: CRPS (Continuous Ranked Probability Score)
"""

import numpy as np
import xarray as xr
from loguru import logger


def crps_ensemble(
    forecasts: xr.DataArray,
    observations: xr.DataArray,
    member_dim: str = "realization",
) -> xr.DataArray:
    """Compute CRPS for ensemble forecasts using the properscoring library.

    CRPS is the primary verification metric — it measures the integrated
    squared difference between the forecast CDF and the observation step function.
    Lower is better. Units match the variable (e.g., Kelvin for temperature).

    Args:
        forecasts: Ensemble forecasts with a member dimension.
        observations: Observations (same grid, no member dimension).
        member_dim: Name of the ensemble member dimension.

    Returns:
        CRPS values (same shape as observations, member dim collapsed).
    """
    import properscoring as ps

    # Align on shared dimensions
    obs_aligned = observations.broadcast_like(
        forecasts.isel({member_dim: 0})
    )

    # properscoring expects (n_members, n_points) arrays
    fc_vals = forecasts.values
    obs_vals = obs_aligned.values

    # Reshape for properscoring
    n_members = fc_vals.shape[forecasts.dims.index(member_dim)]
    member_axis = forecasts.dims.index(member_dim)

    crps_vals = ps.crps_ensemble(
        obs_vals,
        np.moveaxis(fc_vals, member_axis, -1),
    )

    # Wrap back in xarray
    result = obs_aligned.copy(data=crps_vals)
    result.name = "crps"
    result.attrs["units"] = str(forecasts.attrs.get("units", ""))
    result.attrs["long_name"] = "Continuous Ranked Probability Score"

    return result


def crps_skill_score(
    crps_forecast: xr.DataArray,
    crps_reference: xr.DataArray,
) -> xr.DataArray:
    """CRPS Skill Score relative to a reference (typically climatology).

    CRPSS = 1 - CRPS_forecast / CRPS_reference
    CRPSS > 0 means forecast beats reference.
    CRPSS = 1 is perfect.

    Args:
        crps_forecast: CRPS of the forecast being evaluated.
        crps_reference: CRPS of the reference forecast (e.g., climatology).

    Returns:
        Skill score values.
    """
    crpss = 1.0 - crps_forecast / crps_reference
    crpss.name = "crpss"
    crpss.attrs["long_name"] = "CRPS Skill Score"
    return crpss


def brier_score(
    prob_forecast: xr.DataArray,
    binary_obs: xr.DataArray,
) -> xr.DataArray:
    """Brier Score for probability forecasts at a specific threshold.

    BS = mean((forecast_probability - observation_binary)^2)
    Range: [0, 1]. Lower is better.

    Args:
        prob_forecast: Probability forecasts (values in [0, 1]).
        binary_obs: Binary observations (0 or 1).

    Returns:
        Brier Score.
    """
    bs = ((prob_forecast - binary_obs) ** 2).mean()
    bs.name = "brier_score"
    return bs


def reliability_diagram_data(
    prob_forecast: xr.DataArray,
    binary_obs: xr.DataArray,
    n_bins: int = 10,
) -> dict:
    """Compute data for reliability diagrams.

    Groups forecasts into probability bins, computes observed frequency
    in each bin, and returns data for plotting.

    Args:
        prob_forecast: Probability forecasts (values in [0, 1]).
        binary_obs: Binary observations (0 or 1).
        n_bins: Number of probability bins.

    Returns:
        Dict with 'bin_centers', 'observed_freq', 'forecast_freq', 'counts'.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fc_flat = prob_forecast.values.flatten()
    obs_flat = binary_obs.values.flatten()

    # Remove NaNs
    valid = ~(np.isnan(fc_flat) | np.isnan(obs_flat))
    fc_flat = fc_flat[valid]
    obs_flat = obs_flat[valid]

    observed_freq = np.zeros(n_bins)
    forecast_freq = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (fc_flat >= bin_edges[i]) & (fc_flat < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (fc_flat >= bin_edges[i]) & (fc_flat <= bin_edges[i + 1])
        counts[i] = mask.sum()
        if counts[i] > 0:
            observed_freq[i] = obs_flat[mask].mean()
            forecast_freq[i] = fc_flat[mask].mean()

    return {
        "bin_centers": bin_centers,
        "observed_freq": observed_freq,
        "forecast_freq": forecast_freq,
        "counts": counts,
    }


def rank_histogram(
    forecasts: xr.DataArray,
    observations: xr.DataArray,
    member_dim: str = "realization",
) -> np.ndarray:
    """Compute rank histogram (Talagrand diagram) for ensemble calibration.

    The observation is ranked among ensemble members. A well-calibrated
    ensemble produces a uniform rank histogram.

    Args:
        forecasts: Ensemble forecasts.
        observations: Observations.
        member_dim: Ensemble member dimension name.

    Returns:
        Histogram counts for ranks 1 to n_members+1.
    """
    member_axis = forecasts.dims.index(member_dim)
    n_members = forecasts.sizes[member_dim]

    fc = forecasts.values
    obs = np.expand_dims(observations.values, axis=member_axis)

    # Concatenate obs with ensemble and find rank of obs
    combined = np.concatenate([fc, obs], axis=member_axis)
    ranks = np.apply_along_axis(
        lambda x: np.searchsorted(np.sort(x[:-1]), x[-1]) + 1,
        axis=member_axis,
        arr=combined,
    )

    # Flatten and histogram
    ranks_flat = ranks.flatten()
    hist, _ = np.histogram(ranks_flat, bins=np.arange(0.5, n_members + 2.5, 1))

    return hist


def spread_skill_ratio(
    forecasts: xr.DataArray,
    observations: xr.DataArray,
    member_dim: str = "realization",
) -> float:
    """Ensemble spread-skill ratio.

    Ratio of mean ensemble spread to ensemble mean RMSE.
    Ratio = 1.0 indicates perfect calibration.
    Ratio < 1.0 means underdispersive (too confident).
    Ratio > 1.0 means overdispersive.

    Args:
        forecasts: Ensemble forecasts.
        observations: Observations.
        member_dim: Ensemble member dimension name.

    Returns:
        Spread-skill ratio (scalar).
    """
    ens_mean = forecasts.mean(dim=member_dim)
    ens_spread = forecasts.std(dim=member_dim).mean().values
    rmse = float(np.sqrt(((ens_mean - observations) ** 2).mean().values))

    ratio = float(ens_spread / rmse) if rmse > 0 else np.nan
    logger.info(f"Spread-skill ratio: {ratio:.3f} (spread={ens_spread:.3f}, RMSE={rmse:.3f})")
    return ratio


def compute_all_metrics(
    forecasts: xr.DataArray,
    observations: xr.DataArray,
    member_dim: str = "realization",
    threshold_K: float = 273.15,
) -> dict:
    """Compute the full verification suite for one experiment.

    Args:
        forecasts: Ensemble forecasts.
        observations: Observations.
        member_dim: Ensemble member dimension.
        threshold_K: Threshold for Brier Score (Kelvin).

    Returns:
        Dict of metric name → value.
    """
    ens_mean = forecasts.mean(dim=member_dim)

    results = {
        "mae": float(np.abs(ens_mean - observations).mean().values),
        "rmse": float(np.sqrt(((ens_mean - observations) ** 2).mean().values)),
        "bias": float((ens_mean - observations).mean().values),
        "crps_mean": float(crps_ensemble(forecasts, observations, member_dim).mean().values),
        "spread_skill_ratio": spread_skill_ratio(forecasts, observations, member_dim),
    }

    # Brier score at threshold
    prob_fc = (forecasts > threshold_K).mean(dim=member_dim)
    binary_obs = (observations > threshold_K).astype(float)
    results[f"brier_score_{threshold_K:.0f}K"] = float(
        brier_score(prob_fc, binary_obs).values
    )

    logger.info(f"Metrics: MAE={results['mae']:.3f}, RMSE={results['rmse']:.3f}, CRPS={results['crps_mean']:.3f}")
    return results
