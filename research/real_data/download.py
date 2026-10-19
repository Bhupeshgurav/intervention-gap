"""Acquire the authors' real NPZ splits; standard library only, no credentials.

Run from any directory: python research/real_data/download.py
Metadata must first be saved from the official public Dataverse dataset API.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_DIR = ROOT / "data" / "transferable_dynamics"
METADATA = HERE / "sources" / "edmond_metadata.json"
MANIFEST = DATA_DIR / "manifest.json"
DOI = "10.17617/3.ZT6K7P"


def checksums(path: Path) -> dict[str, str]:
    digests = {"md5": hashlib.md5(), "sha256": hashlib.sha256()}
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            for digest in digests.values():
                digest.update(block)
    return {name: digest.hexdigest() for name, digest in digests.items()}


def acquire(item: dict) -> dict:
    filename = item["label"]
    if Path(filename).name != filename:
        raise ValueError("Unsafe source filename")
    if item.get("restricted"):
        raise ValueError(f"Refusing restricted source: {filename}")
    source = item["dataFile"]
    expected_md5 = source["checksum"]["value"]
    if source["checksum"]["type"] != "MD5":
        raise ValueError("Unexpected upstream checksum type")
    target = DATA_DIR / filename
    url = f"https://edmond.mpg.de/api/access/datafile/{source['id']}"
    reuse = target.exists() and checksums(target)["md5"] == expected_md5
    if not reuse:
        partial = target.with_suffix(".npz.part")
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                with partial.open("wb") as stream:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(block)
            actual = checksums(partial)
            if partial.stat().st_size != source["filesize"]:
                raise ValueError(f"Byte count mismatch: {filename}")
            if actual["md5"] != expected_md5:
                raise ValueError(f"Upstream MD5 mismatch: {filename}")
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
    actual = checksums(target)
    if target.stat().st_size != source["filesize"]:
        raise ValueError(f"Byte count mismatch: {filename}")
    print(f"Verified {filename}: {target.stat().st_size:,} bytes", flush=True)
    return {
        "filename": filename,
        "family": filename.split("_", 1)[0],
        "split": filename.split("_", 1)[1].removesuffix(".npz"),
        "source_url": url,
        "dataverse_file_id": source["id"],
        "size_bytes": target.stat().st_size,
        "upstream_md5": expected_md5,
        **actual,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "is_real_measurement": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", choices=("Sines", "GPs"), default=["Sines", "GPs"])
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    metadata = json.loads(METADATA.read_text())["data"]
    version = metadata["latestVersion"]
    selected = [
        item for item in version["files"]
        if item["label"].split("_", 1)[0] in args.families
        and item["label"].endswith(".npz") and not item["label"].endswith("_full.npz")
    ]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(acquire, selected))
    previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    merged = {item["filename"]: item for item in previous.get("files", [])}
    merged.update({item["filename"]: item for item in records})
    result = {
        "dataset_title": "A Real-Robot Dataset for Assessing Transferability of Learned Dynamics Models",
        "doi": DOI,
        "dataset_version": version["versionNumber"],
        "license": version["license"],
        "metadata_source_url": "https://edmond.mpg.de/api/datasets/:persistentId/?persistentId=doi:" + DOI,
        "metadata_sha256": checksums(METADATA)["sha256"],
        "selection": "Official real Sines and GPs split files only; no Sim_Sines or full duplicate arrays.",
        "files": sorted(merged.values(), key=lambda item: item["filename"]),
    }
    MANIFEST.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
