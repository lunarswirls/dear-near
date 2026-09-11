#!/usr/bin/env python3
"""fetch calibrated NEAR MAG data and trajectory kernels from the PDS"""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


start = "2000-01-11"
stop = "2001-02-10"
frame = "nso"

mag_url = "https://sbnarchive.psi.edu/pds4/near/near_mag/data_calibrated"
spice_url = (
    "https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/"
    "nearsp_1000/data"
)

spice_kernels = [
    "lsk/naif0007.tls",
    "spk/eros80.bsp",
    "spk/near_erosorbit_nav_v1.bsp",
]
data_directory = Path("/Users/danywaller/Projects/near/data")


def parse_date(value):
    return datetime.strptime(value, "%Y-%m-%d").date()


def mission_phase(day):
    if date(1998, 12, 20) <= day <= date(1999, 1, 3):
        return "eros_flyby"
    if date(2000, 1, 11) <= day <= date(2001, 2, 10):
        return "eros_orbit"
    raise ValueError(
        f"{day.isoformat()} is outside calibrated Eros flyby/orbit MAG coverage"
    )


def ten_day_directory(day):
    doy = day.timetuple().tm_yday
    first_doy = 1 if doy < 10 else doy // 10 * 10
    return f"{day.year}_{first_doy:03d}"


def download(url, output):
    if output.exists() and output.stat().st_size:
        print(f"exists {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "dear-near/1.0"})
    try:
        with urlopen(request) as response, output.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
    except HTTPError as error:
        output.unlink(missing_ok=True)
        if error.code == 404:
            print(f"missing {url}")
            return
        raise
    print(f"fetched {output}")


def fetch_mag(start, stop, frame, output):
    day = start
    fetched = 0
    while day <= stop:
        phase = mission_phase(day)
        doy = day.timetuple().tm_yday
        filename = f"{frame}{day.year % 100:02d}{doy:03d}"
        directory = ten_day_directory(day)
        remote = f"{mag_url}/{phase}/{directory}"
        local = output / "mag" / phase / directory
        before = (local / f"{filename}.tab").exists()
        download(f"{remote}/{filename}.tab", local / f"{filename}.tab")
        download(f"{remote}/{filename}.xml", local / f"{filename}.xml")
        fetched += int(not before and (local / f"{filename}.tab").exists())
        day += timedelta(days=1)
    return fetched


def fetch_spice(output):
    for relative_path in spice_kernels:
        download(f"{spice_url}/{relative_path}", output / "spice" / relative_path)


start = parse_date(start)
stop = parse_date(stop)
frame = frame.lower()

if start > stop:
    raise ValueError("START must be on or before STOP")
if frame not in {"nso", "ebf"}:
    raise ValueError("FRAME must be nso or ebf")

count = fetch_mag(start, stop, frame, data_directory)
fetch_spice(data_directory)
print(f"downloaded {count} new MAG data files")
print(f"archive checked at {datetime.now(timezone.utc).isoformat()}")
