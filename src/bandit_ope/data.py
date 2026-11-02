"""Download and validate a bounded, integrity-checked Open Bandit sample."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
from datetime import UTC
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = ROOT / "data/source_manifest.json"
RAW_DIR = ROOT / "data/raw"
REPORTS_DIR = ROOT / "reports"
REQUIRED = {"timestamp", "item_id", "position", "click", "propensity_score"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_url(revision: str, path: str) -> str:
    return f"https://huggingface.co/datasets/zozonext/open-bandit/resolve/{revision}/{path}"


def _download_range(url: str, start: int, end: int, source_bytes: int, etag_sha256: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{end}", "If-Match": f'"{etag_sha256}"'},
    )
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                body = response.read()
                content_range = response.headers.get("Content-Range", "")
                response_etag = response.headers.get("ETag", "").strip('"')
                if response.status != 206:
                    raise ValueError(f"Expected HTTP 206 for range, got {response.status}")
                if content_range != f"bytes {start}-{end}/{source_bytes}":
                    raise ValueError(f"Unexpected Content-Range: {content_range}")
                if response_etag != etag_sha256:
                    raise ValueError(f"Source ETag changed: {response_etag!r}")
                if len(body) != end - start + 1:
                    raise ValueError(f"Range length mismatch: received {len(body)} bytes")
                return body
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as error:
            last_error = error
            if attempt == 0:
                time.sleep(1)
    raise RuntimeError(f"Could not fetch verified byte range {start}-{end}: {last_error}")


def _window_bytes(
    *, url: str, source_bytes: int, range_bytes: int, count: int, etag_sha256: str
) -> list[tuple[int, bytes]]:
    max_start = source_bytes - range_bytes
    starts = [round(max_start * index / (count - 1)) for index in range(count)]
    windows: list[tuple[int, bytes]] = []
    for start in starts:
        end = min(start + range_bytes - 1, source_bytes - 1)
        windows.append((start, _download_range(url, start, end, source_bytes, etag_sha256)))
    return windows


def _parse_windows(
    policy: str, windows: list[tuple[int, bytes]], columns: list[str]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    records: list[pd.DataFrame] = []
    details: list[dict[str, Any]] = []
    for window_index, (start, data) in enumerate(windows):
        text = data.decode("utf-8")
        lines = text.splitlines(keepends=True)
        if start == 0:
            lines = lines[1:]
        else:
            lines = lines[1:]
        if lines and not lines[-1].endswith(("\n", "\r")):
            lines = lines[:-1]
        complete = "".join(lines)
        if not complete.strip():
            continue
        frame = pd.read_csv(io.StringIO(complete), header=None, names=columns, low_memory=False)
        frame["_source_policy"] = policy
        frame["_window_index"] = window_index
        frame["_window_start_byte"] = start
        records.append(frame)
        details.append(
            {
                "window_index": window_index,
                "start_byte": start,
                "end_byte_inclusive": start + len(data) - 1,
                "window_sha256": _sha256(data),
                "complete_rows": len(frame),
                "retained_rows": len(frame),
                "first_timestamp": str(frame["timestamp"].iloc[0]),
                "last_timestamp": str(frame["timestamp"].iloc[-1]),
            }
        )
    if not records:
        raise ValueError(f"No complete CSV records parsed for {policy}")
    return pd.concat(records, ignore_index=True), details


def _validate(frame: pd.DataFrame, policy: str) -> dict[str, Any]:
    missing = (REQUIRED | {"source_row_id"}) - set(frame.columns)
    if missing:
        raise ValueError(f"{policy}: required columns missing: {sorted(missing)}")
    if frame.empty or frame["timestamp"].isna().any():
        raise ValueError(f"{policy}: empty sample or missing timestamps")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{policy}: invalid timestamps")
    item_id = pd.to_numeric(frame["item_id"], errors="coerce")
    position = pd.to_numeric(frame["position"], errors="coerce")
    click = pd.to_numeric(frame["click"], errors="coerce")
    propensity = pd.to_numeric(frame["propensity_score"], errors="coerce")
    source_row_id = pd.to_numeric(frame["source_row_id"], errors="coerce")
    if source_row_id.isna().any() or not (source_row_id % 1 == 0).all():
        raise ValueError(f"{policy}: source row IDs must be integers")
    if source_row_id.duplicated().any() or not source_row_id.is_monotonic_increasing:
        raise ValueError(f"{policy}: source row IDs are duplicated or out of order")
    if item_id.isna().any() or not item_id.between(0, 79).all():
        raise ValueError(f"{policy}: item_id outside the 80-arm catalog")
    if position.isna().any() or not position.between(1, 3).all():
        raise ValueError(f"{policy}: position outside 1..3")
    if click.isna().any() or not click.isin([0, 1]).all():
        raise ValueError(f"{policy}: click must be binary")
    if propensity.isna().any() or (propensity <= 0).any() or (propensity > 1).any():
        raise ValueError(f"{policy}: propensity scores must lie in (0, 1]")
    days = timestamps.dt.floor("D")
    unique_days = sorted(days.dt.strftime("%Y-%m-%d").unique().tolist())
    if len(frame) < 10_000:
        raise ValueError(f"{policy}: only {len(frame)} sampled rows; expected at least 10,000")
    if len(unique_days) < 5:
        raise ValueError(f"{policy}: byte sample spans only {len(unique_days)} distinct days")
    return {
        "rows": len(frame),
        "days": unique_days,
        "first_timestamp_utc": timestamps.min().isoformat(),
        "last_timestamp_utc": timestamps.max().isoformat(),
        "click_rate": float(click.mean()),
        "mean_logged_propensity": float(propensity.mean()),
        "item_count": int(item_id.nunique()),
        "position_count": int(position.nunique()),
        "source_row_id_min": int(source_row_id.min()),
        "source_row_id_max": int(source_row_id.max()),
    }


def _header_from_first_window(data: bytes) -> list[str]:
    first_line = data.splitlines()[0].decode("utf-8-sig")
    columns = next(
        pd.read_csv(io.StringIO(first_line), header=None).itertuples(index=False, name=None)
    )
    result = [str(column) for column in columns]
    if result and result[0].strip().lower() in {"nan", "none", ""}:
        result[0] = "source_row_id"
    if not REQUIRED.issubset(result):
        raise ValueError(
            f"Source header is missing required fields: {sorted(REQUIRED - set(result))}"
        )
    if "source_row_id" not in result:
        raise ValueError("Source CSV is missing its expected leading row identifier")
    return result


def fetch_sample() -> dict[str, Any]:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    sampling = manifest["sampling"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "dataset": manifest["dataset"],
        "revision": manifest["revision"],
        "sampling_method": sampling["method"],
        "sampling_note": sampling["note"],
        "policies": {},
    }
    for policy, source in manifest["source_files"].items():
        url = _source_url(manifest["revision"], source["path"])
        windows = _window_bytes(
            url=url,
            source_bytes=source["bytes"],
            range_bytes=sampling["range_bytes"],
            count=sampling["num_ranges_per_file"],
            etag_sha256=source["etag_sha256"],
        )
        columns = _header_from_first_window(windows[0][1])
        frame, details = _parse_windows(policy, windows, columns)
        stats = _validate(frame, policy)
        output = RAW_DIR / f"{policy}_all_sample.csv.gz"
        frame.to_csv(output, index=False, compression="gzip")
        summary["policies"][policy] = {
            **stats,
            "source_path": source["path"],
            "source_bytes": source["bytes"],
            "source_etag_sha256": source["etag_sha256"],
            "sample_file": str(output.relative_to(ROOT)),
            "sample_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "sample_bytes": output.stat().st_size,
            "windows": details,
        }
        print(
            f"{policy}: {stats['rows']:,} rows, {len(stats['days'])} UTC days, "
            f"{stats['click_rate']:.4%} clicks"
        )
    common_days = sorted(
        set(summary["policies"]["bts"]["days"]) & set(summary["policies"]["random"]["days"])
    )
    if len(common_days) < 5:
        raise ValueError(
            f"Policy samples share only {len(common_days)} calendar days: {common_days}"
        )
    summary["common_days"] = common_days
    summary["created_utc"] = pd.Timestamp.now(tz=UTC).isoformat()
    path = REPORTS_DIR / "source_sample_manifest.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote bounded samples and {path.relative_to(ROOT)}")
    return summary


def validate_local() -> dict[str, Any]:
    manifest_path = REPORTS_DIR / "source_sample_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for policy, info in manifest["policies"].items():
        sample = ROOT / info["sample_file"]
        if not sample.exists():
            raise FileNotFoundError(sample)
        if hashlib.sha256(sample.read_bytes()).hexdigest() != info["sample_sha256"]:
            raise ValueError(f"Sample file hash mismatch: {sample}")
        frame = pd.read_csv(sample, compression="gzip")
        current = _validate(frame, policy)
        if current["rows"] != info["rows"] or current["days"] != info["days"]:
            raise ValueError(f"Sample manifest counts or date coverage changed: {policy}")
        print(f"{policy}: validated {current['rows']:,} rows, sha256 verified")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only", action="store_true", help="validate previously downloaded data"
    )
    args = parser.parse_args()
    if args.validate_only:
        validate_local()
    else:
        fetch_sample()


if __name__ == "__main__":
    main()
