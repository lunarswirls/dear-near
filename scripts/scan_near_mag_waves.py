#!/usr/bin/env python3
"""screen the full NEAR MAG record for coherent wave-like signatures"""

from csv import DictWriter
from datetime import datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
from scipy import signal
from scipy.ndimage import median_filter


start = "2000-02-14T15:33:00"
stop = None
window_seconds = 3600.0
overlap = 0.5
minimum_samples = 128
minimum_coverage = 0.8
maximum_gap_factor = 5.0
maximum_resampled_samples = 8192
minimum_peak_ratio = 3.0
minimum_coherence = 0.5
candidate_percentile = 99.0
maximum_candidates = 200

data_directory = Path("/Users/danywaller/Projects/near/data")
output_directory = Path("/Users/danywaller/Projects/near/output/wave_scan")


def parse_time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def read_mag(data_directory, start=None, stop=None):
    files = sorted((data_directory / "mag").rglob("*.tab"))
    if not files:
        raise FileNotFoundError(f"no calibrated MAG tables below {data_directory / 'mag'}")
    frames = {path.name[:3].lower() for path in files}
    if len(frames) != 1 or not frames <= {"nso", "ebf"}:
        raise ValueError(
            "the MAG directory must contain exactly one coordinate frame, nso or ebf"
        )

    time_chunks = []
    field_chunks = []
    for path in files:
        raw_times = np.loadtxt(path, delimiter=",", usecols=0, dtype="U32", ndmin=1)
        fields = np.loadtxt(path, delimiter=",", usecols=(6, 7, 8, 9), ndmin=2)
        times = np.array(
            [np.datetime64(value, "ms") for value in raw_times], dtype="datetime64[ms]"
        )
        time_chunks.append(times)
        field_chunks.append(fields)

    times = np.concatenate(time_chunks)
    fields = np.concatenate(field_chunks)
    order = np.argsort(times)
    times = times[order]
    fields = fields[order]
    keep = np.isfinite(fields).all(axis=1) & (np.abs(fields) < 1e6).all(axis=1)
    if start:
        keep &= times >= np.datetime64(start.replace(tzinfo=None), "ms")
    if stop:
        keep &= times <= np.datetime64(stop.replace(tzinfo=None), "ms")
    times = times[keep]
    fields = fields[keep]
    if times.size < minimum_samples:
        raise ValueError("not enough MAG samples remain in the requested interval")
    return times, fields, frames.pop().upper()


def spectral_features(regular_fields, cadence):
    fluctuations = signal.detrend(regular_fields, axis=0, type="linear")
    vector_amplitude = np.linalg.norm(fluctuations, axis=1)
    rms_fluctuation = float(np.sqrt(np.mean(vector_amplitude**2)))
    mean_field = np.mean(regular_fields, axis=0)
    mean_field_nt = float(np.linalg.norm(mean_field))
    relative_amplitude = rms_fluctuation / max(mean_field_nt, np.finfo(float).eps)
    impulsiveness = float(np.max(vector_amplitude) / max(rms_fluctuation, np.finfo(float).eps))

    covariance = np.cov(fluctuations, rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
    planarity = 1 - eigenvalues[0] / max(eigenvalues[1], np.finfo(float).eps)
    ellipticity = eigenvalues[1] / max(eigenvalues[2], np.finfo(float).eps)

    if mean_field_nt:
        parallel = fluctuations @ (mean_field / mean_field_nt)
        total_variance = max(
            np.sum(np.var(fluctuations, axis=0)), np.finfo(float).eps
        )
        compressibility = float(np.var(parallel) / total_variance)
    else:
        compressibility = np.nan

    sample_count = regular_fields.shape[0]
    nperseg = min(256, max(32, sample_count // 4))
    noverlap = nperseg // 2
    frequencies, component_psd = signal.welch(
        fluctuations,
        fs=1 / cadence,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="linear",
        scaling="density",
        axis=0,
    )
    vector_psd = np.sum(component_psd, axis=1)
    valid = frequencies >= 3 / window_seconds
    valid &= frequencies > 0
    if np.count_nonzero(valid) < 5:
        raise ValueError("analysis window has too few resolved frequencies")

    log_psd = np.log(np.maximum(vector_psd, np.finfo(float).tiny))
    background = median_filter(log_psd, size=9, mode="nearest")
    excess = log_psd - background
    peak_index = np.flatnonzero(valid)[np.argmax(excess[valid])]
    peak_ratio = float(np.exp(excess[peak_index]))
    dominant_frequency = float(frequencies[peak_index])

    normalized_power = vector_psd[valid] / np.sum(vector_psd[valid])
    spectral_entropy = float(
        -np.sum(normalized_power * np.log(normalized_power + np.finfo(float).eps))
        / np.log(normalized_power.size)
    )

    coherence_values = []
    for first, second in [(0, 1), (0, 2), (1, 2)]:
        _, coherence = signal.coherence(
            fluctuations[:, first],
            fluctuations[:, second],
            fs=1 / cadence,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="linear",
        )
        coherence_values.append(coherence[peak_index])
    maximum_coherence = float(np.max(coherence_values))

    artifact_flag = bool(np.max(regular_fields[:, 3]) > 200 or impulsiveness > 12)
    return {
        "mean_field_nt": mean_field_nt,
        "rms_fluctuation_nt": rms_fluctuation,
        "relative_amplitude": relative_amplitude,
        "impulsiveness": impulsiveness,
        "compressibility": compressibility,
        "planarity": float(planarity),
        "ellipticity": float(ellipticity),
        "dominant_frequency_hz": dominant_frequency,
        "peak_ratio": peak_ratio,
        "spectral_entropy": spectral_entropy,
        "maximum_coherence": maximum_coherence,
        "artifact_flag": artifact_flag,
    }


def scan_windows(times, fields):
    seconds = times.astype("datetime64[ms]").astype(np.int64) / 1000
    step_seconds = window_seconds * (1 - overlap)
    window_starts = np.arange(
        seconds[0], seconds[-1] - window_seconds + step_seconds, step_seconds
    )
    records = []

    for window_index, window_start in enumerate(window_starts):
        window_stop = window_start + window_seconds
        left = np.searchsorted(seconds, window_start, side="left")
        right = np.searchsorted(seconds, window_stop, side="left")
        source_times = seconds[left:right]
        source_fields = fields[left:right]
        if source_times.size < minimum_samples:
            continue

        unique_times, unique_indices = np.unique(source_times, return_index=True)
        source_fields = source_fields[unique_indices]
        steps = np.diff(unique_times)
        cadence = float(np.median(steps[steps > 0]))
        cadence = max(cadence, window_seconds / maximum_resampled_samples)
        expected_samples = int(np.floor(window_seconds / cadence))
        if expected_samples < minimum_samples:
            continue
        coverage = unique_times.size / expected_samples
        if coverage < minimum_coverage or np.max(steps) > cadence * maximum_gap_factor:
            continue

        regular_times = window_start + np.arange(expected_samples) * cadence
        regular_fields = np.column_stack(
            [
                np.interp(regular_times, unique_times, source_fields[:, index])
                for index in range(4)
            ]
        )
        try:
            features = spectral_features(regular_fields, cadence)
        except (ValueError, np.linalg.LinAlgError):
            continue
        records.append(
            {
                "start_unix_s": float(window_start),
                "stop_unix_s": float(window_stop),
                "center_unix_s": float(window_start + window_seconds / 2),
                "sample_count": int(unique_times.size),
                "cadence_s": cadence,
                "coverage": float(coverage),
                **features,
            }
        )
        if window_index and window_index % 2000 == 0:
            print(f"scanned {window_index:,} of {window_starts.size:,} windows")

    if not records:
        raise ValueError("no complete, sufficiently gap-free analysis windows were found")
    return records


def load_spice(spice_directory):
    kernels = sorted(spice_directory.rglob("*.tls")) + sorted(
        spice_directory.rglob("*.bsp")
    )
    if not kernels:
        raise FileNotFoundError(f"no SPICE kernels below {spice_directory}")
    spice.kclear()
    for kernel in kernels:
        spice.furnsh(str(kernel))


def add_distances(records, spice_directory):
    load_spice(spice_directory)
    utc = [
        datetime.fromtimestamp(record["center_unix_s"], timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%f"
        )
        for record in records
    ]
    ephemeris_times = np.asarray(spice.str2et(utc))
    positions, _ = spice.spkpos(
        "-93", ephemeris_times, "J2000", "NONE", "2000433"
    )
    spice.kclear()
    distances = np.linalg.norm(positions, axis=1)
    for record, distance in zip(records, distances):
        record["distance_km"] = float(distance)


def robust_zscore(values):
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    deviation = np.nanmedian(np.abs(values - median)) * 1.4826
    if not np.isfinite(deviation) or deviation == 0:
        deviation = np.nanstd(values)
    if not np.isfinite(deviation) or deviation == 0:
        return np.zeros(values.size)
    return (values - median) / deviation


def score_records(records):
    peak_score = robust_zscore(np.log([record["peak_ratio"] for record in records]))
    coherence_score = robust_zscore(
        [record["maximum_coherence"] for record in records]
    )
    entropy_score = robust_zscore(
        [-record["spectral_entropy"] for record in records]
    )
    amplitude_score = robust_zscore(
        np.log([record["relative_amplitude"] for record in records])
    )
    proximity_score = robust_zscore(
        -np.log([record["distance_km"] for record in records])
    )
    wave_scores = (
        0.35 * peak_score
        + 0.25 * coherence_score
        + 0.20 * entropy_score
        + 0.20 * amplitude_score
    )
    interaction_scores = 0.8 * wave_scores + 0.2 * proximity_score

    for index, record in enumerate(records):
        record["wave_score"] = float(wave_scores[index])
        record["proximity_score"] = float(proximity_score[index])
        record["interaction_score"] = float(interaction_scores[index])

    eligible = np.array(
        [
            not record["artifact_flag"]
            and record["peak_ratio"] >= minimum_peak_ratio
            and record["maximum_coherence"] >= minimum_coherence
            for record in records
        ]
    )
    if not np.any(eligible):
        return []
    threshold = np.nanpercentile(interaction_scores[eligible], candidate_percentile)
    candidates = [
        record
        for record, keep in zip(records, eligible)
        if keep and record["interaction_score"] >= threshold
    ]
    candidates.sort(key=lambda record: record["interaction_score"], reverse=True)
    return candidates[:maximum_candidates]


def utc_string(unix_seconds):
    return datetime.fromtimestamp(unix_seconds, timezone.utc).isoformat()


def write_records(path, records):
    fieldnames = [
        "start_utc",
        "stop_utc",
        "center_utc",
        "sample_count",
        "cadence_s",
        "coverage",
        "distance_km",
        "mean_field_nt",
        "rms_fluctuation_nt",
        "relative_amplitude",
        "dominant_frequency_hz",
        "peak_ratio",
        "maximum_coherence",
        "spectral_entropy",
        "compressibility",
        "planarity",
        "ellipticity",
        "impulsiveness",
        "artifact_flag",
        "wave_score",
        "proximity_score",
        "interaction_score",
    ]
    with path.open("w", newline="") as stream:
        writer = DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in fieldnames if key in record}
            row["start_utc"] = utc_string(record["start_unix_s"])
            row["stop_utc"] = utc_string(record["stop_unix_s"])
            row["center_utc"] = utc_string(record["center_unix_s"])
            writer.writerow(row)


def plot_overview(path, records, candidates, frame):
    centers = np.array(
        [record["center_unix_s"] * 1000 for record in records], dtype=np.int64
    ).astype("datetime64[ms]").astype(datetime)
    interaction_scores = np.array(
        [record["interaction_score"] for record in records]
    )
    frequencies = np.array(
        [record["dominant_frequency_hz"] for record in records]
    )
    distances = np.array([record["distance_km"] for record in records])
    peak_ratios = np.array([record["peak_ratio"] for record in records])
    candidate_times = np.array(
        [record["center_unix_s"] * 1000 for record in candidates], dtype=np.int64
    ).astype("datetime64[ms]").astype(datetime)
    candidate_scores = np.array(
        [record["interaction_score"] for record in candidates]
    )

    figure, axes = plt.subplots(4, 1, figsize=(14, 12), constrained_layout=True)
    axes[0].plot(centers, interaction_scores, color="#555555", linewidth=0.5)
    if candidates:
        axes[0].scatter(candidate_times, candidate_scores, color="#d62728", s=18)
    axes[0].set_ylabel("interaction score")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(centers, frequencies, c=np.log10(peak_ratios), s=5, cmap="viridis")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("peak frequency (Hz)")
    axes[1].grid(alpha=0.25)

    axes[2].plot(centers, distances, color="#6a3d9a", linewidth=0.6)
    axes[2].set_ylabel("NEAR–Eros (km)")
    axes[2].grid(alpha=0.25)

    scatter = axes[3].scatter(
        distances,
        interaction_scores,
        c=np.log10(frequencies),
        s=7,
        alpha=0.7,
        cmap="plasma",
    )
    axes[3].set_xlabel("NEAR–Eros distance (km)")
    axes[3].set_ylabel("interaction score")
    axes[3].grid(alpha=0.25)
    figure.colorbar(scatter, ax=axes[3], label="log10 peak frequency (Hz)")
    figure.suptitle(
        f"NEAR MAG {frame} wave-signature screening\n"
        "red points are ranked candidates; artifact-like windows are excluded"
    )
    figure.savefig(path, dpi=200)


if not 0 <= overlap < 1:
    raise ValueError("overlap must be from zero to less than one")
if not 0 < minimum_coverage <= 1:
    raise ValueError("minimum_coverage must be greater than zero and at most one")
if not 0 <= candidate_percentile <= 100:
    raise ValueError("candidate_percentile must be from zero to 100")

start = parse_time(start) if start else None
stop = parse_time(stop) if stop else None
times, fields, frame = read_mag(data_directory, start, stop)
print(f"loaded {times.size:,} {frame} MAG samples")
records = scan_windows(times, fields)
print(f"retained {len(records):,} complete analysis windows")
add_distances(records, data_directory / "spice")
candidates = score_records(records)

output_directory.mkdir(parents=True, exist_ok=True)
write_records(output_directory / "wave_windows.csv", records)
write_records(output_directory / "wave_candidates.csv", candidates)
plot_overview(
    output_directory / "wave_signature_scan.png", records, candidates, frame
)
print(f"ranked {len(candidates):,} candidate windows")
print(f"saved results below {output_directory}")
