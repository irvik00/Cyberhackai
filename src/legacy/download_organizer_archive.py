"""Download and verify the public organizer ZIP; supports resumable transfers."""

import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PUBLIC_URL = "https://disk.yandex.ru/d/WhQO9dUapBEeOA"
API = "https://cloud-api.yandex.net/v1/disk/public/resources"
ROOT = Path(__file__).resolve().parent


def json_request(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def sha256_file(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def main():
    query = urllib.parse.urlencode({"public_key": PUBLIC_URL})
    meta = json_request(API + "?" + query)
    destination = ROOT / "data" / "organizer_pages.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".zip.part")
    expected_size = int(meta["size"])
    expected_hash = meta["sha256"]
    if destination.exists():
        if destination.stat().st_size == expected_size and sha256_file(destination) == expected_hash:
            print("Organizer archive already downloaded and verified", flush=True)
            return
        raise RuntimeError("Existing archive differs from organizer metadata")

    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise RuntimeError("Partial file is larger than the official archive")
    if offset < expected_size:
        link = json_request(API + "/download?" + query)["href"]
        request = urllib.request.Request(link, headers={"Range": f"bytes={offset}-"} if offset else {})
        with urllib.request.urlopen(request, timeout=60) as response:
            if offset and response.status != 206:
                raise RuntimeError("Server did not honor the resume range")
            with partial.open("ab" if offset else "wb") as target:
                milestone = offset
                while block := response.read(2 * 1024 * 1024):
                    target.write(block)
                    offset += len(block)
                    if offset - milestone >= 64 * 1024 * 1024:
                        print(f"Download: {offset / 1024**2:.0f} / {expected_size / 1024**2:.0f} MiB", flush=True)
                        milestone = offset
    if partial.stat().st_size != expected_size:
        raise RuntimeError("Incomplete archive; rerun to resume")
    actual_hash = sha256_file(partial)
    if actual_hash != expected_hash:
        raise RuntimeError("Archive SHA-256 mismatch")
    partial.replace(destination)
    (destination.parent / "organizer_source.json").write_text(
        json.dumps({"public_url": PUBLIC_URL, "size": expected_size, "sha256": actual_hash}, indent=2), encoding="utf-8"
    )
    print(f"Verified organizer archive: {destination.name}; SHA-256 {actual_hash}", flush=True)


if __name__ == "__main__":
    main()
