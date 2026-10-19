"""Download pinned public URL datasets and verify their exact archive hashes."""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SOURCES = [
    {"file": "malicious_urls_v1.zip", "source_page": "https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset",
     "download_url": "https://www.kaggle.com/api/v1/datasets/download/sid321axn/malicious-urls-dataset?datasetVersionNumber=1",
     "license": "CC0: Public Domain", "creator": "Manu Siddhartha", "version": 1,
     "bytes": 17748901, "sha256": "8d085ce1fb4e462111e233a6a302970f699264ae129e3f6e5ac0496f5b69ca79"},
    {"file": "url_dataset_v1.zip", "source_page": "https://www.kaggle.com/datasets/teseract/urldataset",
     "download_url": "https://www.kaggle.com/api/v1/datasets/download/teseract/urldataset?datasetVersionNumber=1",
     "license": "CC0: Public Domain", "creator": "TeseRact", "version": 1,
     "bytes": 6911526, "sha256": "d68d9dcb133aa3190ea758ff9dd1e044bdcbd0c643b415b076ca2fdf3f5f0048"},
    {"file": "phiusiil_uci_967.zip", "source_page": "https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset",
     "download_url": "https://archive.ics.uci.edu/static/public/967/phiusiil%2Bphishing%2Burl%2Bdataset.zip",
     "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
     "citation": "Prasad, A. & Chandra, S. (2024). PhiUSIIL Phishing URL (Website) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.1016/j.cose.2023.103545.",
     "bytes": 15376007, "sha256": "0a639fd03aea6308c5b1c10c92aa23c2ce1505447a9137271865cd0badc9a59a"},
]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    folder = ROOT / "data" / "external"
    folder.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        path = folder / source["file"]
        if not path.exists():
            temp = path.with_suffix(".zip.part")
            request = urllib.request.Request(source["download_url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response, temp.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            if temp.stat().st_size != source["bytes"] or sha256_file(temp) != source["sha256"]:
                raise ValueError("External source changed or download is incomplete")
            temp.replace(path)
        if path.stat().st_size != source["bytes"] or sha256_file(path) != source["sha256"]:
            raise ValueError(f"External archive differs from the fixed version: {path.name}")
        print(f"Verified external data: {path.name}", flush=True)
    (folder / "sources.json").write_text(json.dumps(SOURCES, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
