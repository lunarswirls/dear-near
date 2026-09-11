#!/usr/bin/env python3
"""calculate sliding-window power spectral densities for NEAR MAG data"""

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


start = "2000-02-14T00:00:00"
stop = "2000-02-15T00:00:00"
window_seconds = 3600.0
overlap = 0.5

data_directory = Path("/Users/danywaller/Projects/near/data")
output_stem = Path("/Users/danywaller/Projects/near/output/eros_psd")


def parse_time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def read_mag(data_directory, start=None, stop=None):
    root = data_directory / "mag" if (data_directory / "mag").exists() else data_directory
    files = sorted(root.rglob("*.tab"))
    if not files:
        raise FileNotFoundError(f"no calibrated MAG tables below {root}")
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
    keep = np.ones(times.size, dtype=bool)
    if start:
        keep &= times >= np.datetime64(start.replace(tzinfo=None), "ms")
    if stop:
        keep &= times <= np.datetime64(stop.replace(tzinfo=None), "ms")
    times = times[keep]
    fields = fields[keep]
    if times.size < 4:
        raise ValueError("at least four MAG samples are required")
    return times, fields, frames.pop().upper()


def windowed_psd(times, fields, window_seconds, overlap):
    seconds = times.astype("datetime64[ms]").astype(np.int64) / 1000
    positive_steps = np.diff(seconds)
    positive_steps = positive_steps[positive_steps > 0]
    cadence = float(np.median(positive_steps))
    samples_per_window = int(round(window_seconds / cadence))
    if samples_per_window < 8:
        raise ValueError("WINDOW_SECONDS must span at least eight samples")
    step_seconds = window_seconds * (1 - overlap)
    starts = np.arange(seconds[0], seconds[-1] - window_seconds + cadence, step_seconds)
    frequencies = np.fft.rfftfreq(samples_per_window, cadence)
    spectra = []
    centers = []

    for window_start in starts:
        window_stop = window_start + window_seconds
        keep = (seconds >= window_start) & (seconds < window_stop)
        source_times = seconds[keep]
        source_fields = fields[keep]
        if source_times.size < samples_per_window * 0.8:
            continue
        unique_times, unique_indices = np.unique(source_times, return_index=True)
        source_fields = source_fields[unique_indices]
        if unique_times.size < 4 or np.max(np.diff(unique_times)) > cadence * 5:
            continue
        regular_times = window_start + np.arange(samples_per_window) * cadence
        regular_fields = np.column_stack(
            [
                np.interp(regular_times, unique_times, source_fields[:, index])
                for index in range(4)
            ]
        )
        component_spectra = []
        for index in range(4):
            _, power = signal.periodogram(
                regular_fields[:, index],
                fs=1 / cadence,
                window="hann",
                detrend="linear",
                scaling="density",
            )
            component_spectra.append(power)
        spectra.append(np.column_stack(component_spectra))
        centers.append(window_start + window_seconds / 2)

    if not spectra:
        raise ValueError("no complete, sufficiently gap-free analysis windows were found")
    return frequencies, np.asarray(centers), np.stack(spectra), cadence


start = parse_time(start) if start else None
stop = parse_time(stop) if stop else None

if window_seconds <= 0:
    raise ValueError("WINDOW_SECONDS must be positive")
if not 0 <= overlap < 1:
    raise ValueError("OVERLAP must be from 0 to less than 1")

times, fields, frame = read_mag(data_directory, start, stop)
frequencies, centers, spectra, cadence = windowed_psd(
    times, fields, window_seconds, overlap
)

stem = output_stem.with_suffix("")
figure_path = stem.with_suffix(".png")
data_path = stem.with_suffix(".npz")
figure_path.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    data_path,
    frequency_hz=frequencies,
    window_center_unix_s=centers,
    psd_bx=spectra[:, :, 0],
    psd_by=spectra[:, :, 1],
    psd_bz=spectra[:, :, 2],
    psd_btotal=spectra[:, :, 3],
    cadence_s=cadence,
    window_seconds=window_seconds,
    overlap=overlap,
    coordinate_frame=frame,
)

plot_times = np.array(centers * 1000, dtype="datetime64[ms]").astype(datetime)
positive = frequencies > 0
labels = [r"$B_x$", r"$B_y$", r"$B_z$", r"$|B|$"]
figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
for index, axis in enumerate(axes):
    mesh = axis.pcolormesh(
        plot_times,
        frequencies[positive],
        spectra[:, positive, index].T,
        shading="auto",
        norm="log",
        cmap="viridis",
    )
    axis.set_yscale("log")
    axis.set_ylabel(f"{labels[index]}\nfrequency (Hz)")
    figure.colorbar(mesh, ax=axis, label=r"PSD (nT$^2$/Hz)")

axes[-1].set_xlabel("window center (UTC)")
axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator()))
figure.suptitle(
    f"NEAR MAG {frame} sliding-window PSD "
    f"({window_seconds:g} s, {overlap:.0%} overlap)"
)
figure.savefig(figure_path, dpi=200)
print(f"saved {data_path}")
print(f"saved {figure_path}")
