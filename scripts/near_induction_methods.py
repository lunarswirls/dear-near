"""data-driven induction tests for the close-orbit NEAR MAG record"""

import numpy as np
from scipy.ndimage import uniform_filter1d


def prepare_close_bins(times, values, settings):
    seconds = times.astype("datetime64[ms]").astype(np.int64) / 1000
    field = values[:, 1:4]
    btotal = values[:, 4]
    position = values[:, 5:8]
    distance = np.linalg.norm(position, axis=1)
    field_magnitude = np.linalg.norm(field, axis=1)
    tolerance = np.maximum(
        settings["btotal_absolute_tolerance_nt"],
        settings["btotal_relative_tolerance"] * np.maximum(btotal, 1),
    )
    selected = distance < settings["maximum_distance_km"]
    selected &= np.abs(field_magnitude - btotal) <= tolerance
    selected &= np.isfinite(field).all(axis=1)
    selected &= np.isfinite(position).all(axis=1)
    seconds = seconds[selected]
    field = field[selected]
    btotal = btotal[selected]
    position = position[selected]
    if seconds.size == 0:
        raise ValueError("no quality-controlled samples are strictly inside the limit")

    bin_seconds = settings["bin_seconds"]
    bin_number = np.floor(seconds / bin_seconds).astype(np.int64)
    _, first, counts = np.unique(bin_number, return_index=True, return_counts=True)
    bin_time = np.empty(first.size)
    bin_field = np.empty((first.size, 3))
    bin_position = np.empty((first.size, 3))
    bin_btotal = np.empty(first.size)
    for index, (start, count) in enumerate(zip(first, counts)):
        stop = start + count
        bin_time[index] = np.median(seconds[start:stop])
        bin_field[index] = np.median(field[start:stop], axis=0)
        bin_position[index] = np.median(position[start:stop], axis=0)
        bin_btotal[index] = np.median(btotal[start:stop])

    distance = np.linalg.norm(bin_position, axis=1)
    strict = distance < settings["maximum_distance_km"]
    bin_time = bin_time[strict]
    bin_field = bin_field[strict]
    bin_position = bin_position[strict]
    bin_btotal = bin_btotal[strict]
    counts = counts[strict]
    gaps = np.diff(bin_time)
    block = np.r_[0, np.cumsum(gaps > settings["block_gap_seconds"])]
    block_values, block_counts = np.unique(block, return_counts=True)
    retained_blocks = block_values[
        block_counts >= settings["minimum_bins_per_block"]
    ]
    retained = np.isin(block, retained_blocks)
    block = block[retained]
    _, block = np.unique(block, return_inverse=True)
    return {
        "time_unix_s": bin_time[retained],
        "field_nt": bin_field[retained],
        "btotal_nt": bin_btotal[retained],
        "position_km": bin_position[retained],
        "distance_km": distance[retained],
        "raw_sample_count": counts[retained],
        "block": block,
        "bin_seconds": float(bin_seconds),
        "raw_selected_count": int(np.count_nonzero(selected)),
    }


def smooth_by_block(values, block, window_bins):
    smoothed = np.empty_like(values, dtype=float)
    for block_value in np.unique(block):
        indices = np.flatnonzero(block == block_value)
        size = min(int(window_bins), indices.size)
        if size % 2 == 0:
            size -= 1
        if size < 3:
            smoothed[indices] = np.mean(values[indices], axis=0)
        else:
            smoothed[indices] = uniform_filter1d(
                values[indices],
                size=size,
                axis=0,
                mode="nearest",
            )
    return smoothed


def dipole_tensor(position_km, reference_radius_km):
    distance = np.linalg.norm(position_km, axis=1)
    radial = position_km / distance[:, None]
    identity = np.eye(3)[None, :, :]
    geometry = 3 * radial[:, :, None] * radial[:, None, :] - identity
    return geometry * (reference_radius_km / distance)[:, None, None] ** 3


def residualize_blocks(values, design, block):
    centered_values = values.copy()
    centered_design = design.copy()
    for block_value in np.unique(block):
        selected = block == block_value
        centered_values[selected] -= np.mean(values[selected], axis=0)
        centered_design[selected] -= np.mean(design[selected], axis=0)
    return centered_values, centered_design


def model_arrays(data, background_seconds, reference_radius_km, lag_seconds=0):
    window_bins = max(3, round(background_seconds / data["bin_seconds"]))
    external = smooth_by_block(data["field_nt"], data["block"], window_bins)
    if lag_seconds:
        lag_bins = int(round(lag_seconds / data["bin_seconds"]))
        lagged = np.empty_like(external)
        for block_value in np.unique(data["block"]):
            indices = np.flatnonzero(data["block"] == block_value)
            source = np.clip(
                np.arange(indices.size) - lag_bins,
                0,
                indices.size - 1,
            )
            lagged[indices] = external[indices[source]]
        external = lagged
    residual = data["field_nt"] - smooth_by_block(
        data["field_nt"],
        data["block"],
        window_bins,
    )
    geometry = dipole_tensor(data["position_km"], reference_radius_km)
    induced_scalar = np.einsum("nij,nj->ni", geometry, external)
    induced_tensor = np.einsum(
        "nca,nb->ncab",
        geometry,
        external,
    ).reshape(-1, 3, 9)
    permanent_design = geometry
    return {
        "residual_nt": residual,
        "external_nt": external,
        "geometry": geometry,
        "permanent_design": permanent_design,
        "scalar_design": np.concatenate(
            (permanent_design, induced_scalar[:, :, None]),
            axis=2,
        ),
        "tensor_design": np.concatenate(
            (permanent_design, induced_tensor),
            axis=2,
        ),
    }


def block_sufficient_statistics(values, design, block):
    values, design = residualize_blocks(values, design, block)
    parameter_count = design.shape[2]
    statistics = []
    for block_value in np.unique(block):
        selected = block == block_value
        selected_values = values[selected].reshape(-1)
        selected_design = design[selected].reshape(-1, parameter_count)
        statistics.append(
            {
                "xtx": selected_design.T @ selected_design,
                "xty": selected_design.T @ selected_values,
                "yty": float(selected_values @ selected_values),
            }
        )
    return statistics


def solve_statistics(statistics, ridge=1e-10):
    xtx = np.sum([item["xtx"] for item in statistics], axis=0)
    xty = np.sum([item["xty"] for item in statistics], axis=0)
    scale = np.trace(xtx) / max(xtx.shape[0], 1)
    regularization = np.eye(xtx.shape[0]) * max(scale * ridge, ridge)
    return np.linalg.solve(xtx + regularization, xty)


def cross_validated_r2(values, design, block):
    statistics = block_sufficient_statistics(values, design, block)
    total_xtx = np.sum([item["xtx"] for item in statistics], axis=0)
    total_xty = np.sum([item["xty"] for item in statistics], axis=0)
    total_yty = sum(item["yty"] for item in statistics)
    residual_sum = 0.0
    for held_out in statistics:
        xtx = total_xtx - held_out["xtx"]
        xty = total_xty - held_out["xty"]
        scale = np.trace(xtx) / max(xtx.shape[0], 1)
        regularization = np.eye(xtx.shape[0]) * max(scale * 1e-10, 1e-10)
        coefficients = np.linalg.solve(xtx + regularization, xty)
        residual_sum += held_out["yty"]
        residual_sum -= 2 * coefficients @ held_out["xty"]
        residual_sum += coefficients @ held_out["xtx"] @ coefficients
    return 1 - residual_sum / max(total_yty, np.finfo(float).tiny)


def fit_induction_models(data, background_seconds, reference_radius_km):
    arrays = model_arrays(data, background_seconds, reference_radius_km)
    permanent_r2 = cross_validated_r2(
        arrays["residual_nt"],
        arrays["permanent_design"],
        data["block"],
    )
    scalar_r2 = cross_validated_r2(
        arrays["residual_nt"],
        arrays["scalar_design"],
        data["block"],
    )
    tensor_r2 = cross_validated_r2(
        arrays["residual_nt"],
        arrays["tensor_design"],
        data["block"],
    )
    scalar_statistics = block_sufficient_statistics(
        arrays["residual_nt"],
        arrays["scalar_design"],
        data["block"],
    )
    scalar_coefficients = solve_statistics(scalar_statistics)
    return {
        "bin_seconds": data["bin_seconds"],
        "background_seconds": float(background_seconds),
        "bin_count": data["time_unix_s"].size,
        "block_count": np.unique(data["block"]).size,
        "permanent_cv_r2": permanent_r2,
        "scalar_cv_r2": scalar_r2,
        "tensor_cv_r2": tensor_r2,
        "scalar_delta_cv_r2": scalar_r2 - permanent_r2,
        "tensor_delta_cv_r2": tensor_r2 - permanent_r2,
        "permanent_moment_nt": scalar_coefficients[:3],
        "scalar_response": float(scalar_coefficients[-1]),
    }


def shifted_geometry(data, generator):
    shifted = data["position_km"].copy()
    for block_value in np.unique(data["block"]):
        indices = np.flatnonzero(data["block"] == block_value)
        if indices.size > 1:
            shifted[indices] = np.roll(
                shifted[indices],
                generator.integers(1, indices.size),
                axis=0,
            )
    return shifted


def position_shift_test(
    data,
    background_seconds,
    reference_radius_km,
    draws,
    generator,
):
    arrays = model_arrays(data, background_seconds, reference_radius_km)
    observed_permanent = cross_validated_r2(
        arrays["residual_nt"], arrays["permanent_design"], data["block"]
    )
    observed_scalar = cross_validated_r2(
        arrays["residual_nt"], arrays["scalar_design"], data["block"]
    )
    observed = observed_scalar - observed_permanent
    null = np.empty(draws)
    for draw in range(draws):
        geometry = dipole_tensor(
            shifted_geometry(data, generator),
            reference_radius_km,
        )
        induced = np.einsum("nij,nj->ni", geometry, arrays["external_nt"])
        permanent_design = geometry
        scalar_design = np.concatenate(
            (permanent_design, induced[:, :, None]),
            axis=2,
        )
        permanent_r2 = cross_validated_r2(
            arrays["residual_nt"], permanent_design, data["block"]
        )
        scalar_r2 = cross_validated_r2(
            arrays["residual_nt"], scalar_design, data["block"]
        )
        null[draw] = scalar_r2 - permanent_r2
    pvalue = (np.count_nonzero(null >= observed) + 1) / (draws + 1)
    return {
        "observed_delta_cv_r2": observed,
        "null": null,
        "null_lower": float(np.percentile(null, 2.5)),
        "null_upper": float(np.percentile(null, 97.5)),
        "pvalue": pvalue,
    }


def block_bootstrap_response(
    data,
    background_seconds,
    reference_radius_km,
    draws,
    generator,
):
    arrays = model_arrays(data, background_seconds, reference_radius_km)
    statistics = block_sufficient_statistics(
        arrays["residual_nt"], arrays["scalar_design"], data["block"]
    )
    responses = np.empty(draws)
    for draw in range(draws):
        indices = generator.integers(0, len(statistics), len(statistics))
        selected = [statistics[index] for index in indices]
        responses[draw] = solve_statistics(selected)[-1]
    return responses


def scan_lags(
    data,
    background_seconds,
    reference_radius_km,
    lag_seconds,
):
    result = []
    for lag in lag_seconds:
        arrays = model_arrays(
            data,
            background_seconds,
            reference_radius_km,
            lag_seconds=lag,
        )
        permanent_r2 = cross_validated_r2(
            arrays["residual_nt"], arrays["permanent_design"], data["block"]
        )
        scalar_r2 = cross_validated_r2(
            arrays["residual_nt"], arrays["scalar_design"], data["block"]
        )
        result.append(scalar_r2 - permanent_r2)
    return np.asarray(result)
