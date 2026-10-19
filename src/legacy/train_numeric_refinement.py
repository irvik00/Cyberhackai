"""Train a nonlinear model on numeric URL/page features for comparison."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import gzip
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from train_model import URLStats, read_inputs

ROOT = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BRANDS = ("paypal", "apple", "microsoft", "google", "facebook", "amazon", "outlook", "office", "netflix", "yahoo", "ebay", "icloud", "bank")
TOKENS = ("login", "password", "verify", "account", "secure", "signin", "confirm", "update")


def entropy(value):
    counts = Counter(value)
    n = max(len(value), 1)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())


def extra_features(row):
    url, title, text = row["url"].lower(), row["title"].lower(), row["text"].lower()
    try:
        parsed = urlsplit(url if "://" in url else "//" + url)
        host, path, query = parsed.hostname or "", parsed.path or "", parsed.query or ""
    except ValueError:
        host = path = query = ""
    return [
        entropy(host), sum(c.isdigit() for c in host) / max(len(host), 1),
        host.count("-"), max((len(t) for t in host.split(".")), default=0),
        math.log1p(len(title.split())), entropy(title),
        sum(ord(c) > 127 for c in title) / max(len(title), 1),
        len(set(row["tags"].split())), math.log1p(len(set(row["link_hosts"].split()))),
        int(bool(re.search(r"(?:url|redirect|target|continue|next|q)=", query))),
        sum(brand in title and brand not in host for brand in BRANDS),
        *[int(token in url) for token in TOKENS],
        *[int(token in text) for token in TOKENS],
        *[int(host.endswith("." + ending)) for ending in ("com", "org", "net", "ru", "br", "uk", "in", "xyz", "info", "top", "online", "tk", "cf", "ml", "ga", "gq")],
        *[int(path.endswith(ending)) for ending in (".php", ".html", ".htm", ".asp", ".aspx", ".pdf")],
    ]


def numeric_matrix(name, wanted_urls):
    values, urls = [], []
    with gzip.open(ROOT / "content_features" / f"{name}.jsonl.gz", "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["url"] in wanted_urls:
                urls.append(row["url"])
                values.append(row["numeric"] + extra_features(row))
    return urls, np.hstack([URLStats().transform(urls).toarray(), np.asarray(values, dtype=np.float32)])


def model():
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=2.0, class_weight="balanced",
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=42,
    )


def main():
    output = ROOT / "experiments" / "numeric_refinement"
    output.mkdir(parents=True, exist_ok=True)
    _, _, urls, y, test_urls, *_ = read_inputs(ROOT / "train_6dd2c989.csv", ROOT / "test_5013633d.csv")
    train_order, features = numeric_matrix("train", set(urls))
    test_order, test_features = numeric_matrix("test", set(test_urls))
    assert train_order == urls.tolist() and test_order == test_urls.tolist()
    saved = np.load(ROOT / "experiments/url_html/model_scores.npz")
    assert np.array_equal(y, saved["y"])
    folds, scores = saved["fold_ids"], np.empty(len(y), dtype=np.float32)
    print(f"Numeric feature matrix: {features.shape}", flush=True)
    iterations = []
    for fold in range(1, 6):
        fit, valid = folds != fold, folds == fold
        fitted = model().fit(features[fit], y[fit])
        scores[valid] = fitted.predict_proba(features[valid])[:, 1]
        iterations.append(int(fitted.n_iter_))
        print(f"Numeric model fold {fold}/5: {fitted.n_iter_} iterations", flush=True)
    final = model().fit(features, y)
    test_scores = final.predict_proba(test_features)[:, 1]
    np.savez_compressed(output / "model_scores.npz", y=y, fold_ids=folds, oof=scores, test=test_scores)
    (output / "report.json").write_text(json.dumps({"features": int(features.shape[1]), "fold_iterations": iterations, "final_iterations": int(final.n_iter_), "parameters": final.get_params()}, indent=2), encoding="utf-8")
    print("Numeric OOF and test scores saved", flush=True)


if __name__ == "__main__":
    main()
