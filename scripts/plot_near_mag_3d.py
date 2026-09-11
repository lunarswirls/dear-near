#!/usr/bin/env python3
"""plot all NEAR MAG measurements over the Eros orbits in three dimensions"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import spiceypy as spice


orbit_insertion = "2000-02-14T15:33:00"
start = None
stop = None
color_component = "btotal"
maximum_points = 100000
gap_seconds = 1800.0
marker_size = 3.0

data_directory = Path("/Users/danywaller/Projects/near/data")
output = Path("/Users/danywaller/Projects/near/output/eros_orbits_3d.html")

components = {"bx": 0, "by": 1, "bz": 2, "btotal": 3}
labels = {
    "bx": "Bx (nT)",
    "by": "By (nT)",
    "bz": "Bz (nT)",
    "btotal": "|B| (nT)",
}


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
    keep = np.ones(times.size, dtype=bool)
    if start:
        keep &= times >= np.datetime64(start.replace(tzinfo=None), "ms")
    if stop:
        keep &= times <= np.datetime64(stop.replace(tzinfo=None), "ms")
    if not np.any(keep):
        raise ValueError("no MAG samples fall inside the requested interval")
    return times[keep], fields[keep], frames.pop().upper()


def downsample(times, fields, maximum_points):
    if times.size <= maximum_points:
        return times, fields
    indices = np.linspace(0, times.size - 1, maximum_points, dtype=int)
    indices = np.unique(indices)
    return times[indices], fields[indices]


def load_spice(spice_directory):
    kernels = sorted(spice_directory.rglob("*.tls")) + sorted(
        spice_directory.rglob("*.bsp")
    )
    if not kernels:
        raise FileNotFoundError(f"no SPICE kernels below {spice_directory}")
    spice.kclear()
    for kernel in kernels:
        spice.furnsh(str(kernel))


def eros_positions(times, spice_directory):
    load_spice(spice_directory)
    utc = [np.datetime_as_string(value, unit="ms") for value in times]
    positions = []
    for first in range(0, times.size, 10000):
        ephemeris_times = np.asarray(spice.str2et(utc[first : first + 10000]))
        chunk, _ = spice.spkpos(
            "-93", ephemeris_times, "J2000", "NONE", "2000433"
        )
        positions.append(chunk)
    spice.kclear()
    return np.concatenate(positions)


def trajectory_with_gaps(times, positions, gap_seconds):
    seconds = times.astype("datetime64[ms]").astype(np.int64) / 1000
    breaks = np.flatnonzero(np.diff(seconds) > gap_seconds) + 1
    x = positions[:, 0].astype(object)
    y = positions[:, 1].astype(object)
    z = positions[:, 2].astype(object)
    for index in breaks[::-1]:
        x = np.insert(x, index, None)
        y = np.insert(y, index, None)
        z = np.insert(z, index, None)
    return x, y, z


if color_component not in components:
    raise ValueError(f"color_component must be one of {', '.join(components)}")
if maximum_points < 2:
    raise ValueError("maximum_points must be at least two")
if gap_seconds <= 0:
    raise ValueError("gap_seconds must be positive")

orbit_insertion = parse_time(orbit_insertion)
start = max(parse_time(start), orbit_insertion) if start else orbit_insertion
stop = parse_time(stop) if stop else None
times, fields, frame = read_mag(data_directory, start, stop)
times, fields = downsample(times, fields, maximum_points)
positions = eros_positions(times, data_directory / "spice")
line_x, line_y, line_z = trajectory_with_gaps(times, positions, gap_seconds)
distance = np.linalg.norm(positions, axis=1)
color = fields[:, components[color_component]]
utc = [np.datetime_as_string(value, unit="ms") for value in times]
hover = [
    "<br>".join(
        [
            f"UTC: {utc[index]}",
            f"Bx: {fields[index, 0]:.3f} nT",
            f"By: {fields[index, 1]:.3f} nT",
            f"Bz: {fields[index, 2]:.3f} nT",
            f"|B|: {fields[index, 3]:.3f} nT",
            f"distance: {distance[index]:.3f} km",
        ]
    )
    for index in range(times.size)
]

figure = go.Figure()
figure.add_trace(
    go.Scatter3d(
        x=line_x,
        y=line_y,
        z=line_z,
        mode="lines",
        line={"color": "rgba(70, 70, 70, 0.45)", "width": 1},
        hoverinfo="skip",
        name="NEAR trajectory",
    )
)
figure.add_trace(
    go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="markers",
        marker={
            "size": marker_size,
            "color": color,
            "colorscale": "Turbo",
            "colorbar": {"title": labels[color_component]},
            "opacity": 0.85,
        },
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        name="MAG samples",
    )
)
figure.add_trace(
    go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode="markers",
        marker={"size": 7, "color": "#222222", "symbol": "diamond"},
        hovertemplate="Eros<extra></extra>",
        name="Eros",
    )
)
figure.update_layout(
    title=f"NEAR MAG over post-injection Eros orbits ({frame} field, J2000 position)",
    template="plotly_white",
    scene={
        "xaxis_title": "J2000 X relative to Eros (km)",
        "yaxis_title": "J2000 Y relative to Eros (km)",
        "zaxis_title": "J2000 Z relative to Eros (km)",
        "aspectmode": "data",
    },
    legend={"orientation": "h", "y": 1.02, "x": 0},
    margin={"l": 0, "r": 0, "b": 0, "t": 70},
)
output.parent.mkdir(parents=True, exist_ok=True)
figure.write_html(output, include_plotlyjs=True, full_html=True)
print(f"plotted {times.size} MAG samples")
print(f"saved {output}")
