"""Evaluate external URL evidence after removing every local training URL."""

import csv
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_external_sources import SOURCES, canonical_url, host
from refine_predictions import evaluate
from train_model import read_inputs

ROOT = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    raw, _, urls, y, test_urls, *_ = read_inputs(ROOT / "train_6dd2c989.csv", ROOT / "test_5013633d.csv")
    training_keys = {canonical_url(value) for value in raw.url.fillna("").astype(str)}
    train_hosts, test_hosts = [host(url) for url in urls], [host(url) for url in test_urls]
    wanted_hosts = set(train_hosts + test_hosts)
    evidence = defaultdict(lambda: [set(), set()])
    source_counts = []
    csv.field_size_limit(1 << 26)
    for archive_name, member, label_name, labels in SOURCES:
        removed, used = 0, 0
        with zipfile.ZipFile(ROOT / "data" / "external" / archive_name) as archive:
            with archive.open(member) as binary, io.TextIOWrapper(binary, encoding="utf-8-sig", errors="replace", newline="") as text:
                for row in csv.DictReader(text):
                    label = labels.get(row[label_name])
                    if label is None:
                        continue
                    url = row.get("url", row.get("URL", ""))
                    key = canonical_url(url)
                    if key in training_keys:
                        removed += 1
                        continue
                    hostname = host(url)
                    if hostname in wanted_hosts:
                        evidence[hostname][label].add(key)
                        used += 1
        source_counts.append({"archive": archive_name, "local_training_overlap_removed": removed, "relevant_external_rows": used})
    saved = np.load(ROOT / "experiments/refinements/refined_scores.npz")
    assert np.array_equal(y, saved["y"])
    scores, folds, base_known = saved["oof"], saved["fold_ids"], saved["known"]
    baseline = evaluate(y, scores, base_known, folds)
    results, rule_arrays = [], {}
    for support in (1, 2, 5, 10):
        mapping = {h: int(bool(values[1])) for h, values in evidence.items()
                   if not min(len(values[0]), len(values[1])) and max(len(values[0]), len(values[1])) >= support}
        rule = np.asarray([mapping.get(h, -1) for h in train_hosts], dtype=np.int8)
        test_rule = np.asarray([mapping.get(h, -1) for h in test_hosts], dtype=np.int8)
        rule_arrays[f"train_support_{support}"] = rule
        rule_arrays[f"test_support_{support}"] = test_rule
        for distance in (0.25, 0.5, 1.0, 100.0):
            known = base_known.copy()
            use = (known < 0) & (rule >= 0) & (np.abs(scores - baseline["threshold"]) <= distance)
            known[use] = rule[use]
            results.append({"external_minimum_unique_urls": support, "score_distance": distance,
                            "new_known_rows": int(use.sum()), **evaluate(y, scores, known, folds)})
    results.sort(key=lambda row: row["threshold_other_folds_mean"], reverse=True)
    folder = ROOT / "experiments" / "external"
    folder.mkdir(parents=True, exist_ok=True)
    report = {"protocol": "All URLs matching any local training URL are removed from external evidence before evaluating any fold. Only external host labels without class conflicts are used; previous lookup labels are preserved.",
              "source_counts": source_counts, "baseline": baseline, "candidates": results}
    (folder / "decontaminated_host_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(folder / "host_rules.npz", **rule_arrays)
    print(json.dumps({"sources": source_counts, "baseline": baseline, "best": results[:6]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
