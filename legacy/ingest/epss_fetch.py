from __future__ import annotations

import csv
from typing import Dict, Iterable

import urllib.request


def fetch_epss_csv(url: str = "https://epss.cyentia.com/epss_scores-current.csv.gz") -> bytes:
    with urllib.request.urlopen(url, timeout=30) as resp:  # nosec - controlled URL
        return resp.read()


def parse_epss_csv_gz(data_gz: bytes) -> tuple[Dict[str, float], Dict[str, float]]:
    import gzip
    scores: Dict[str, float] = {}
    percentiles: Dict[str, float] = {}
    with gzip.GzipFile(fileobj=bytes_to_fileobj(data_gz)) as f:
        reader = csv.DictReader((line.decode("utf-8") for line in f))
        for row in reader:
            cve = row.get("cve")
            try:
                score = float(row.get("epss", "0") or 0)
            except ValueError:
                score = 0.0
            try:
                pct = float(row.get("percentile", "0") or 0)
            except ValueError:
                pct = 0.0
            if cve:
                scores[cve] = score
                percentiles[cve] = pct
    return scores, percentiles


def bytes_to_fileobj(b: bytes):
    import io
    return io.BytesIO(b)


