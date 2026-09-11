#!/usr/bin/env python3
"""plot calibrated NEAR MAG data with SPICE Eros distance"""

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice


start = "2000-02-14T00:00:00"
stop = "2000-02-15T00:00:00"

colors = {"bx": "#d55e00", "by": "#009e73", "bz": "#0072b2", "btotal": "#222222"}
data_directory = Path("/Users/danywaller/Projects/near/data")
output = Path("/Users/danywaller/Projects/near/output/eros_mag.png")


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
        times = np.loadtxt(path, delimiter=",", usecols=0, dtype="U32", ndmin=1)
        fields = np.loadtxt(path, delimiter=",", usecols=(6, 7, 8, 9), ndmin=2)
        parsed_times = np.array(
            [np.datetime64(value, "ms") for value in times], dtype="datetime64[ms]"
        )
        time_chunks.append(parsed_times)
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
    if not np.any(keep):
        raise ValueError("no MAG samples fall inside the requested interval")
    return times[keep], fields[keep], frames.pop().upper()


def load_spice(spice_directory):
    kernels = sorted(spice_directory.rglob("*.tls")) + sorted(
        spice_directory.rglob("*.bsp")
    )
    if not kernels:
        raise FileNotFoundError(f"no SPICE kernels below {spice_directory}")
    spice.kclear()
    for kernel in kernels:
        spice.furnsh(str(kernel))


def eros_distance(times, spice_directory):
    load_spice(spice_directory)
    utc = [np.datetime_as_string(value, unit="ms") for value in times]
    ephemeris_times = np.asarray(spice.str2et(utc))
    positions, _ = spice.spkpos("-93", ephemeris_times, "J2000", "NONE", "2000433")
    spice.kclear()
    return np.linalg.norm(positions, axis=1)


start = parse_time(start) if start else None
stop = parse_time(stop) if stop else None

times, fields, frame = read_mag(data_directory, start, stop)
distance = eros_distance(times, data_directory / "spice")
plot_times = times.astype("datetime64[ms]").astype(datetime)

figure, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
labels = [r"$B_x$", r"$B_y$", r"$B_z$", r"$|B|$"]
keys = ["bx", "by", "bz", "btotal"]
for index, (label, key) in enumerate(zip(labels, keys)):
    axes[index].plot(plot_times, fields[:, index], color=colors[key], linewidth=0.7)
    axes[index].set_ylabel(f"{label}\n(nT)")
    axes[index].grid(alpha=0.25)

axes[-1].plot(plot_times, distance, color="#6a3d9a", linewidth=0.9)
axes[-1].set_ylabel("NEAR–Eros\n(km)")
axes[-1].set_xlabel("UTC")
axes[-1].grid(alpha=0.25)
axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator()))
figure.suptitle(f"NEAR MAG ({frame}) and SPICE distance from Eros")
output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(output, dpi=200)
print(f"saved {output}")
