"""Compare template/domain lookup refinements on saved five-fold OOF scores."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from train_model import apply_known_labels, best_f1_threshold, read_inputs

ROOT = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def fingerprint(value, minimum=1):
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if len(value) >= minimum else ""


def domain(host):
    parts = host.split(".")
    count = 3 if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in {"co", "com", "net", "org", "gov", "edu", "ac"} else 2
    return ".".join(parts[-count:]) if len(parts) >= 2 else ""


def canonical_url(url):
    try:
        p = urlsplit(url if "://" in url else "//" + url)
        host = re.sub(r"^www\d*\.", "", p.hostname or "")
        return host + re.sub(r"/+", "/", unquote(p.path)).rstrip("/")
    except ValueError:
        return ""


def load_records(path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return pd.DataFrame([json.loads(line) for line in source])


def feature_keys(frame, hosts):
    result = {name: [] for name in (
        "canonical_host", "domain", "canonical_url", "title", "visible_text",
        "normalized_text", "title_tags", "tags", "tags_links", "text_tags",
    )}
    for row, host in zip(frame.itertuples(index=False), hosts):
        title, text = normalize(row.title), normalize(row.text)
        normalized = re.sub(r"https?://\S+|\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", " REFERENCE ", text)
        normalized = re.sub(re.escape(host), " HOST ", normalized) if host else normalized
        normalized = re.sub(r"\d+", " N ", normalized)
        normalized = " ".join(normalized.split())
        tags = row.tags if len(row.tags.split()) >= 30 else ""
        result["canonical_host"].append(re.sub(r"^www\d*\.", "", host))
        result["domain"].append(domain(host))
        result["canonical_url"].append(canonical_url(row.url))
        result["title"].append(fingerprint(title, 12))
        result["visible_text"].append(fingerprint(text, 200))
        result["normalized_text"].append(fingerprint(normalized, 200))
        result["title_tags"].append(fingerprint(title + "|" + tags) if len(title) >= 8 and tags else "")
        result["tags"].append(fingerprint(tags) if tags else "")
        result["tags_links"].append(fingerprint(tags + "|" + row.link_hosts) if tags and row.link_hosts else "")
        result["text_tags"].append(fingerprint(text + "|" + tags) if len(text) >= 40 and tags else "")
    return {name: np.asarray(values, dtype=object) for name, values in result.items()}


def label_map(keys, y, minimum=1):
    observed = {}
    for key, target in zip(keys, y):
        if key:
            counts = observed.setdefault(key, [0, 0])
            counts[int(target)] += 1
    return {key: int(counts[1] > 0) for key, counts in observed.items()
            if min(counts) == 0 and sum(counts) >= minimum}


def lookup(keys, mapping):
    return np.asarray([mapping.get(key, -1) for key in keys], dtype=np.int8)


def oof_lookup(keys, y, folds, minimum):
    prediction = np.full(len(y), -1, dtype=np.int8)
    for fold in range(1, 6):
        fit, valid = folds != fold, folds == fold
        prediction[valid] = lookup(keys[valid], label_map(keys[fit], y[fit], minimum))
    return prediction


def evaluate(y, scores, known, folds):
    threshold, _ = best_f1_threshold(y, scores, known)
    predicted = apply_known_labels(scores, known, threshold)
    stable = []
    for fold in range(1, 6):
        fit, valid = folds != fold, folds == fold
        selected, _ = best_f1_threshold(y[fit], scores[fit], known[fit])
        stable.append(float(f1_score(y[valid], apply_known_labels(scores[valid], known[valid], selected))))
    return {"oof_f1": float(f1_score(y, predicted)), "threshold": float(threshold),
            "fold_f1": [float(f1_score(y[folds == fold], predicted[folds == fold])) for fold in range(1, 6)],
            "threshold_other_folds_f1": stable, "threshold_other_folds_mean": float(np.mean(stable)),
            "errors": int((predicted != y).sum()), "known_rows": int((known >= 0).sum()),
            "known_errors": int(((known >= 0) & (known != y)).sum())}


def main():
    output = ROOT / "experiments" / "refinements"
    output.mkdir(parents=True, exist_ok=True)
    _, _, urls, y, test_urls, hosts, test_hosts, *_ = read_inputs(ROOT / "train_6dd2c989.csv", ROOT / "test_5013633d.csv")
    train = load_records(ROOT / "content_features" / "train.jsonl.gz")
    train = train[train.url.isin(set(urls))].reset_index(drop=True)
    assert train.url.tolist() == urls.tolist()
    saved = np.load(ROOT / "experiments/url_html/model_scores.npz")
    assert np.array_equal(y, saved["y"])
    folds, scores, base_known = saved["fold_ids"], saved["html_oof"], saved["html_known"]
    keys = feature_keys(train, hosts)
    baseline = evaluate(y, scores, base_known, folds)
    results, rule_predictions = [], {}
    for name, values in keys.items():
        for minimum in (1, 2, 3):
            rule = oof_lookup(values, y, folds, minimum)
            rule_predictions[f"{name}:{minimum}"] = rule
            for priority in ("after", "before"):
                known = base_known.copy()
                use = (rule >= 0) & ((known < 0) if priority == "after" else True)
                known[use] = rule[use]
                result = {"rule": name, "minimum": minimum, "priority": priority,
                          "changed_known_rows": int(((known != base_known) & (known >= 0)).sum()),
                          **evaluate(y, scores, known, folds)}
                results.append(result)
            print(f"Evaluated {name}, support >= {minimum}", flush=True)
    results.sort(key=lambda row: row["threshold_other_folds_mean"], reverse=True)
    report = {"baseline": baseline, "single_rules": results,
              "description": "Each rule map uses only labels in the fitting part of its original fold. Before replaces any previous lookup; after applies only to previously unknown rows."}
    (output / "lookup_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(output / "lookup_oof.npz", **rule_predictions)
    print(json.dumps({"baseline": baseline, "best_rules": results[:15]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
