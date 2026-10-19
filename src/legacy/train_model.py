"""Train two URL-only phishing classifiers and write competition submissions.

Expected input files in the current directory:
  train.csv: columns ``url`` and ``result`` (0 = safe, 1 = phishing)
  test.csv:  column ``url``

The validation threshold is selected from out-of-fold decision scores to
maximize F1. Submissions preserve the row order of test.csv.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


class URLStats(BaseEstimator, TransformerMixin):
    """Small set of scale-normalized URL structure features."""

    _digit_run = re.compile(r"\d+")
    _ipv4 = re.compile(r"(?:\d{1,3}\.){3}\d{1,3}")
    _tokens = re.compile(r"[A-Za-z0-9-]+")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for value in X:
            raw = "" if pd.isna(value) else str(value).strip()
            candidate = raw if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", raw) else "//" + raw.lstrip("/")
            try:
                parsed = urlsplit(candidate)
                host = parsed.hostname or ""
                path = parsed.path or ""
                query = parsed.query or ""
                fragment = parsed.fragment or ""
            except ValueError:
                host = path = query = fragment = ""

            n = max(len(raw), 1)
            digits = sum(ch.isdigit() for ch in raw)
            alpha = sum(ch.isalpha() for ch in raw)
            dots_in_host = host.count(".")
            params = query.count("&") + (1 if query else 0)
            runs = self._digit_run.findall(raw)
            rows.append(
                [
                    math.log1p(len(raw)),
                    math.log1p(len(host)),
                    math.log1p(len(path)),
                    math.log1p(len(query)),
                    math.log1p(len(fragment)),
                    math.log1p(dots_in_host),
                    math.log1p(params),
                    digits / n,
                    alpha / n,
                    sum(ch.isupper() for ch in raw) / n,
                    sum(ch not in string.ascii_letters + string.digits for ch in raw) / n,
                    sum(ord(ch) > 127 for ch in raw) / n,
                    raw.count("."),
                    raw.count("-"),
                    raw.count("_"),
                    raw.count("/"),
                    raw.count("@"),
                    raw.count("?"),
                    raw.count("&"),
                    raw.count("="),
                    raw.count("%"),
                    raw.count(":"),
                    raw.count("//"),
                    max((len(run) for run in runs), default=0),
                    len(self._tokens.findall(raw)),
                    float(bool(self._ipv4.search(host))),
                    float(raw.lower().startswith("https://")),
                    float(raw.lower().startswith("http://")),
                ]
            )
        return csr_matrix(np.asarray(rows, dtype=np.float32))


def url_host(value: str) -> str:
    raw = "" if pd.isna(value) else str(value).strip()
    candidate = raw if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", raw) else "//" + raw.lstrip("/")
    try:
        return urlsplit(candidate).hostname or ""
    except ValueError:
        return ""


def unanimous_label_map(keys, labels):
    observed = {}
    for key, label in zip(keys, labels):
        if not key:
            continue
        observed.setdefault(key, set()).add(int(label))
    return {key: next(iter(values)) for key, values in observed.items() if len(values) == 1}


def make_char_model(c: float = 1.0) -> Pipeline:
    features = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=300_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    return Pipeline(
        [
            ("tfidf", features),
            ("classifier", LinearSVC(C=c, class_weight="balanced", max_iter=10_000, dual="auto", random_state=42)),
        ]
    )


def make_hybrid_model(c: float = 1.0) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    min_df=2,
                    max_features=300_000,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    token_pattern=r"(?u)[A-Za-z0-9_]+",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            ("stats", Pipeline([("extract", URLStats()), ("scale", StandardScaler(with_mean=False))])),
        ],
        transformer_weights={"char": 1.0, "word": 0.6, "stats": 0.15},
    )
    return Pipeline(
        [
            ("features", features),
            ("classifier", LinearSVC(C=c, class_weight="balanced", max_iter=10_000, dual="auto", random_state=42)),
        ]
    )


def apply_known_labels(scores, known_labels, threshold):
    predictions = (scores >= threshold).astype(np.int8)
    known = known_labels >= 0
    predictions[known] = known_labels[known]
    return predictions


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray, known_labels: np.ndarray) -> tuple[float, float]:
    unknown = known_labels < 0
    if not unknown.any():
        return 0.0, float(f1_score(y_true, known_labels))

    y_known = y_true[~unknown]
    p_known = known_labels[~unknown]
    fixed_tp = int(((y_known == 1) & (p_known == 1)).sum())
    fixed_fp = int(((y_known == 0) & (p_known == 1)).sum())
    fixed_fn = int(((y_known == 1) & (p_known == 0)).sum())

    y_unknown = y_true[unknown]
    scores_unknown = scores[unknown]
    precision, recall, thresholds = precision_recall_curve(y_unknown, scores_unknown)
    if thresholds.size == 0:
        return 0.0, float(f1_score(y_true, apply_known_labels(scores, known_labels, 0.0)))

    positives = max(int(y_unknown.sum()), 1)
    tp = recall[:-1] * positives
    fp = tp * (1.0 / np.maximum(precision[:-1], 1e-12) - 1.0)
    fn = positives - tp
    total_f1 = 2.0 * (tp + fixed_tp) / np.maximum(
        2.0 * (tp + fixed_tp) + fp + fixed_fp + fn + fixed_fn,
        1e-12,
    )
    best = int(np.nanargmax(total_f1))
    return float(thresholds[best]), float(total_f1[best])


def out_of_fold_scores(model, urls, hosts, labels, folds, seed, model_name):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = np.empty(len(labels), dtype=np.float32)
    known_labels = np.full(len(labels), -1, dtype=np.int8)
    for fold, (fit_idx, valid_idx) in enumerate(cv.split(urls, labels), start=1):
        fold_model = clone(model)
        fold_model.fit(urls[fit_idx], labels[fit_idx])
        scores[valid_idx] = fold_model.decision_function(urls[valid_idx])

        exact_map = unanimous_label_map(urls[fit_idx], labels[fit_idx])
        host_map = unanimous_label_map(hosts[fit_idx], labels[fit_idx])
        for idx in valid_idx:
            if urls[idx] in exact_map:
                known_labels[idx] = exact_map[urls[idx]]
            elif hosts[idx] in host_map:
                known_labels[idx] = host_map[hosts[idx]]
        print(
            f"[{model_name}] fold {fold}/{folds}: {int((known_labels[valid_idx] >= 0).sum())} "
            "validation rows use unanimous train labels",
            flush=True,
        )
    return scores, known_labels


def read_inputs(train_path: Path, test_path: Path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    required_train = {"url", "result"}
    if not required_train.issubset(train.columns):
        raise ValueError(f"{train_path} must contain columns: url, result; found {list(train.columns)}")
    if "url" not in test.columns:
        raise ValueError(f"{test_path} must contain a url column; found {list(test.columns)}")

    all_urls = train["url"].fillna("").astype(str).to_numpy()
    test_urls = test["url"].fillna("").astype(str).to_numpy()
    all_labels = pd.to_numeric(train["result"], errors="raise").astype(int).to_numpy()
    if set(np.unique(all_labels)) != {0, 1}:
        raise ValueError("result must contain both classes encoded as 0 and 1")

    labels_by_url = {}
    conflicting_urls = set()
    for url, label in zip(all_urls, all_labels):
        previous = labels_by_url.get(url)
        if previous is not None and previous != int(label):
            conflicting_urls.add(url)
        else:
            labels_by_url[url] = int(label)
    for url in conflicting_urls:
        labels_by_url.pop(url, None)

    usable = np.fromiter((url not in conflicting_urls for url in all_urls), dtype=bool)
    urls = all_urls[usable]
    labels = all_labels[usable]
    hosts = np.asarray([url_host(url) for url in urls], dtype=object)
    test_hosts = np.asarray([url_host(url) for url in test_urls], dtype=object)
    labels_by_host = unanimous_label_map(hosts, labels)
    audit = {
        "ambiguous_exact_urls_excluded": len(conflicting_urls),
        "ambiguous_training_rows_excluded": int((~usable).sum()),
        "exact_train_test_matches_with_unambiguous_label": sum(url in labels_by_url for url in set(test_urls)),
        "test_rows_with_unanimous_train_host_label": int(sum(host in labels_by_host for host in test_hosts)),
    }
    return train, test, urls, labels, test_urls, hosts, test_hosts, labels_by_url, labels_by_host, audit


def evaluate_and_submit(name, model, urls, hosts, labels, test_urls, test_hosts, test_rows, exact_labels, host_labels, folds, seed, output_dir):
    print(f"[{name}] generating {folds}-fold out-of-fold scores...", flush=True)
    oof_scores, oof_known = out_of_fold_scores(model, urls, hosts, labels, folds, seed, name)
    threshold, oof_f1 = best_f1_threshold(labels, oof_scores, oof_known)
    oof_default_f1 = float(f1_score(labels, apply_known_labels(oof_scores, oof_known, 0.0)))

    print(f"[{name}] fitting final model...", flush=True)
    model.fit(urls, labels)
    test_scores = model.decision_function(test_urls)
    known_labels = np.full(test_rows, -1, dtype=np.int8)
    exact_known = np.fromiter((url in exact_labels for url in test_urls), dtype=bool)
    host_known = np.fromiter((host in host_labels for host in test_hosts), dtype=bool) & ~exact_known
    for i, url in enumerate(test_urls):
        if exact_known[i]:
            known_labels[i] = exact_labels[url]
        elif host_known[i]:
            known_labels[i] = host_labels[test_hosts[i]]
    predictions = apply_known_labels(test_scores, known_labels, threshold)

    output_path = output_dir / f"submission_{name}.csv"
    pd.DataFrame({"Id": np.arange(test_rows), "Predicted": predictions}).to_csv(output_path, index=False)
    report = {
        "model": name,
        "C": float(model.named_steps["classifier"].C),
        "folds": folds,
        "seed": seed,
        "oof_f1_at_threshold": oof_f1,
        "oof_f1_at_zero_threshold": oof_default_f1,
        "decision_threshold": threshold,
        "predicted_phishing": int(predictions.sum()),
        "test_rows": int(test_rows),
        "oof_unanimous_label_rows": int((oof_known >= 0).sum()),
        "exact_label_overrides": int(exact_known.sum()),
        "host_label_overrides": int(host_known.sum()),
        "submission": output_path.name,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument("--out", type=Path, default=Path("."))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=("both", "char", "hybrid"), default="both")
    parser.add_argument("--char-c", type=float, default=1.0)
    parser.add_argument("--hybrid-c", type=float, default=1.0)
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")

    train, test, urls, labels, test_urls, hosts, test_hosts, exact_labels, host_labels, audit = read_inputs(args.train, args.test)
    print(f"Training rows used: {len(urls)} of {len(train)}; test rows: {len(test)}")
    print(f"Class counts: {json.dumps({int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))})}")
    print(f"Exact duplicate train URLs: {int(pd.Series(urls).duplicated().sum())}")
    print(f"Data audit: {json.dumps(audit, ensure_ascii=False)}")
    args.out.mkdir(parents=True, exist_ok=True)

    reports = []
    if args.model in ("both", "char"):
        reports.append(evaluate_and_submit("char_svm", make_char_model(args.char_c), urls, hosts, labels, test_urls, test_hosts, len(test), exact_labels, host_labels, args.folds, args.seed, args.out))
    if args.model in ("both", "hybrid"):
        reports.append(evaluate_and_submit("hybrid_svm", make_hybrid_model(args.hybrid_c), urls, hosts, labels, test_urls, test_hosts, len(test), exact_labels, host_labels, args.folds, args.seed, args.out))
    with (args.out / "cv_report.json").open("w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
