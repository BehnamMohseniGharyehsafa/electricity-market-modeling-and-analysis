"""Download and validate official NYISO P-2A monthly archives, without API keys."""

from __future__ import annotations

import hashlib
import io
import json
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ZONES = [
    "CAPITL",
    "CENTRL",
    "DUNWOD",
    "GENESE",
    "HUD VL",
    "LONGIL",
    "MHK VL",
    "MILLWD",
    "N.Y.C.",
    "NORTH",
    "WEST",
]
TARGET = ZONES.index("N.Y.C.")
PRICE = "LBMP ($/MWHr)"
BASE = "https://mis.nyiso.com/public/csv/damlbmp"


def archive_url(month: pd.Timestamp) -> str:
    return f"{BASE}/{month:%Y%m}01damlbmp_zone_csv.zip"


def fetch(start: str, end: str, root: Path) -> list[Path]:
    """Inclusive monthly archive retrieval; cache and fingerprint exact received bytes."""
    raw = root / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    paths, manifest = [], []
    for month in pd.date_range(start, end, freq="MS"):
        url = archive_url(month)
        path = raw / url.rsplit("/", 1)[1]
        if not path.exists():
            for attempt in range(4):
                try:
                    request = urllib.request.Request(
                        url, headers={"User-Agent": "ElectricityMarketLab/1.0"}
                    )
                    with urllib.request.urlopen(request, timeout=90) as response:
                        payload = response.read()
                    # Never cache an HTML error page or a corrupt ZIP as valid data.
                    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                        if archive.testzip() is not None:
                            raise ValueError(f"Corrupt archive: {url}")
                    path.write_bytes(payload)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2**attempt)
            time.sleep(0.15)
        payload = path.read_bytes()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            files = [n for n in archive.namelist() if n.endswith("damlbmp_zone.csv")]
            if len(files) != month.days_in_month:
                raise ValueError(f"Missing daily files in {path}: {len(files)}")
        paths.append(path)
        manifest.append(
            {
                "url": url,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "daily_files": len(files),
            }
        )
        print(f"Validated {path.name}", flush=True)
    report = root / "reports"
    report.mkdir(exist_ok=True)
    (report / "data_manifest.json").write_text(
        json.dumps(
            {
                "source": "NYISO P-2A day-ahead zonal LBMP",
                "unit": "USD/MWh",
                "manifest_created_utc": datetime.now(timezone.utc).isoformat(),
                "note": "Archive corrections are included; this is not an as-of vintage archive.",
                "archives": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def parse_daily(frame: pd.DataFrame) -> tuple[np.ndarray, bool, int]:
    """Return a canonical 24x11 curve and preserve whether the source day had 24 hours.

    DST is resolved by *calendar curves*, not by pretending local timestamps are UTC.
    Historical repeated hours are averaged; the missing spring hour is interpolated.
    Only history is normalized: a non-24-hour target is excluded by build_samples.
    """
    frame = frame.loc[frame["Name"].isin(ZONES)].copy()
    if frame.empty:
        raise ValueError("No internal NYISO zones found")
    stamp = pd.to_datetime(frame["Time Stamp"], format="%m/%d/%Y %H:%M")
    if stamp.dt.normalize().nunique() != 1:
        raise ValueError("A daily file must contain exactly one market date")
    frame["hour"] = stamp.dt.hour
    values = pd.to_numeric(frame[PRICE], errors="raise").to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Non-finite electricity price")
    counts = frame.groupby("Name").size().reindex(ZONES)
    if counts.isna().any() or counts.nunique() != 1:
        raise ValueError("Incomplete or inconsistent zone coverage")
    n_hours = int(counts.iloc[0])
    # Require the exact clock-hour multiplicities for this local calendar date.
    day = stamp.iloc[0].normalize()
    expected = pd.date_range(
        day, day + pd.Timedelta(days=1), freq="h", inclusive="left", tz="America/New_York"
    )
    expected_counts = pd.Series(expected.hour).value_counts().sort_index()
    for zone in ZONES:
        observed = frame.loc[frame["Name"] == zone, "hour"].value_counts().sort_index()
        if not observed.equals(expected_counts):
            raise ValueError(f"Unexpected hour multiplicity for {day.date()} / {zone}")
    curve = frame.pivot_table(index="hour", columns="Name", values=PRICE, aggfunc="mean")
    curve = curve.reindex(index=range(24), columns=ZONES).interpolate(limit_area="inside")
    if curve.isna().any().any():
        raise ValueError("Unresolved missing hours or zones")
    return curve.to_numpy(dtype=np.float32), n_hours == 24, len(frame)


def prepare(paths: list[Path], root: Path) -> Path:
    curves, normal, days, row_count = [], [], [], 0
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if not name.endswith("damlbmp_zone.csv"):
                    continue
                frame = pd.read_csv(io.BytesIO(archive.read(name)))
                curve, regular, count = parse_daily(frame)
                curves.append(curve)
                normal.append(regular)
                days.append(pd.to_datetime(name[:8], format="%Y%m%d"))
                row_count += count
    dates = pd.DatetimeIndex(days)
    if dates.has_duplicates or not dates.equals(pd.date_range(dates.min(), dates.max(), freq="D")):
        raise ValueError("Daily archive coverage is not unique and continuous")
    target = root / "data/processed/nyiso_daily.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        curves=np.stack(curves),
        normal=np.array(normal),
        dates=dates.to_numpy(dtype="datetime64[D]"),
    )
    summary = {
        "first_day": str(dates.min().date()),
        "last_day": str(dates.max().date()),
        "days": len(days),
        "zones": ZONES,
        "internal_zone_rows": row_count,
        "non_24_hour_days": [str(dates[i].date()) for i, v in enumerate(normal) if not v],
        "min_price": float(np.min(curves)),
        "max_price": float(np.max(curves)),
        "negative_price_cells": int((np.array(curves) < 0).sum()),
    }
    (root / "reports/data_quality.json").write_text(json.dumps(summary, indent=2) + "\n")
    return target
