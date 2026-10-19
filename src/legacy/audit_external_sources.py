"""Audit independent CC0 URL datasets without using test ground truth."""

import csv
import html
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SOURCES = [
    ("malicious_urls_v1.zip", "malicious_phish.csv", "type", {"benign": 0, "phishing": 1}),
    ("url_dataset_v1.zip", "urldata.csv", "label", {"good": 0, "bad": 1}),
    ("phiusiil_uci_967.zip", "PhiUSIIL_Phishing_URL_Dataset.csv", "label", {"1": 0, "0": 1}),
]


def canonical_url(value):
    raw = html.unescape(str(value).strip())
    try:
        parsed = urlsplit(raw if "://" in raw else "http://" + raw.lstrip("/"))
        host = (parsed.hostname or "").lower().removeprefix("www.")
        port = f":{parsed.port}" if parsed.port and parsed.port not in (80, 443) else ""
        return host + port + parsed.path.rstrip("/") + ("?" + parsed.query if parsed.query else "") + ("#" + parsed.fragment if parsed.fragment else "")
    except ValueError:
        return raw


def host(value):
    try:
        raw = str(value).strip()
        return (urlsplit(raw if "://" in raw else "http://" + raw.lstrip("/")).hostname or "").lower().removeprefix("www.")
    except ValueError:
        return ""


def main():
    csv.field_size_limit(1 << 26)
    with (ROOT / "train_6dd2c989.csv").open(encoding="utf-8-sig", newline="") as stream:
        train = list(csv.DictReader(stream))
    with (ROOT / "test_5013633d.csv").open(encoding="utf-8-sig", newline="") as stream:
        test = list(csv.DictReader(stream))
    wanted = {canonical_url(row["url"]) for row in train + test}
    hosts = {host(row["url"]) for row in train + test}
    matches, all_hosts = defaultdict(dict), defaultdict(dict)
    source_reports = []
    for archive_name, member, label_name, labels in SOURCES:
        exact, host_counts, counts = defaultdict(set), defaultdict(lambda: [0, 0]), Counter()
        with zipfile.ZipFile(ROOT / "data" / "external" / archive_name) as archive:
            with archive.open(member) as binary, io.TextIOWrapper(binary, encoding="utf-8-sig", errors="replace", newline="") as text:
                for row in csv.DictReader(text):
                    label = labels.get(row[label_name])
                    if label is None:
                        continue
                    counts["used_rows"] += 1
                    value = row.get("url", row.get("URL", ""))
                    key = canonical_url(value)
                    if key in wanted:
                        exact[key].add(label)
                    h = host(value)
                    if h in hosts:
                        host_counts[h][label] += 1
        pure = {key: next(iter(labels)) for key, labels in exact.items() if len(labels) == 1}
        h_pure = {h: int(n[1] > 0) for h, n in host_counts.items() if min(n) == 0}
        matched_train = [(row, pure[canonical_url(row["url"])]) for row in train if canonical_url(row["url"]) in pure]
        wrong = [{"url": row["url"], "train_label": int(row["result"]), "external_label": value} for row, value in matched_train if int(row["result"]) != value]
        report = {
            "archive": archive_name, "used_rows": counts["used_rows"],
            "exact_train_matches": len(matched_train), "exact_train_disagreements": len(wrong),
            "exact_test_matches": sum(canonical_url(row["url"]) in pure for row in test),
            "test_rows_with_external_unanimous_host": sum(host(row["url"]) in h_pure for row in test),
            "train_disagreement_examples": wrong[:30],
        }
        source_reports.append(report)
        for key, value in pure.items():
            matches[key][archive_name] = value
        for h, count in host_counts.items():
            all_hosts[h][archive_name] = count
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    folder = ROOT / "experiments" / "external"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "audit.json").write_text(json.dumps(source_reports, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "matching_labels.json").write_text(json.dumps(matches, ensure_ascii=False), encoding="utf-8")
    (folder / "matching_host_counts.json").write_text(json.dumps(all_hosts, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
