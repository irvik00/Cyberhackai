"""Paired URL baseline / URL+HTML cross-validation and reproducible submissions."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from train_model import apply_known_labels, best_f1_threshold, make_hybrid_model, read_inputs, unanimous_label_map

ROOT = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
WEIGHTS = {"url": 1.0, "host": 0.7, "title": 0.5, "text": 0.6, "link_hosts": 0.3, "tags": 0.25, "html_numeric": 0.15}


def sha256_file(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def load_frame(path, numeric_names):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        records = [json.loads(line) for line in source]
    frame = pd.DataFrame(records)
    del records
    numeric = np.asarray(frame.pop("numeric").tolist(), dtype=np.float32)
    for idx, column in enumerate(numeric_names):
        frame[column] = numeric[:, idx]
    for column in ("url", "title", "text", "link_hosts", "tags", "html_sha256"):
        frame[column] = frame[column].fillna("").astype(str)
    return frame


def components():
    return {
        "url": make_hybrid_model().named_steps["features"],
        "host": TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=120_000, sublinear_tf=True, dtype=np.float32),
        "title": TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=70_000, sublinear_tf=True, dtype=np.float32),
        "text": TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=180_000, sublinear_tf=True, strip_accents="unicode", dtype=np.float32),
        "link_hosts": TfidfVectorizer(token_pattern=r"(?u)[A-Za-z0-9_]+", ngram_range=(1, 2), min_df=3, max_features=60_000, sublinear_tf=True, dtype=np.float32),
        "tags": TfidfVectorizer(token_pattern=r"[a-z]+", ngram_range=(1, 3), min_df=3, max_features=50_000, sublinear_tf=True, dtype=np.float32),
        "html_numeric": StandardScaler(with_mean=False),
    }


def branch_values(frame, key, numeric_names):
    if key == "html_numeric":
        return frame[numeric_names].to_numpy(dtype=np.float32)
    return frame[key].to_numpy()


def fit_features(fit, validation, numeric_names, log_prefix):
    fit_blocks, valid_blocks = {}, {}
    fitted = components()
    for key, transformer in fitted.items():
        fit_blocks[key] = csr_matrix(transformer.fit_transform(branch_values(fit, key, numeric_names)), dtype=np.float32)
        valid_blocks[key] = csr_matrix(transformer.transform(branch_values(validation, key, numeric_names)), dtype=np.float32)
        print(f"{log_prefix} {key}: {fit_blocks[key].shape[1]} features", flush=True)
    return fitted, fit_blocks, valid_blocks


def combined(blocks):
    return hstack([blocks[key] * weight for key, weight in WEIGHTS.items()], format="csr", dtype=np.float32)


def classifier():
    return LinearSVC(C=1.0, class_weight="balanced", max_iter=10_000, dual="auto", random_state=42)


def known_maps(frame, y):
    return {
        "url": unanimous_label_map(frame["url"].to_numpy(), y),
        "host": unanimous_label_map(frame["host"].to_numpy(), y),
        "html_sha256": unanimous_label_map(frame["html_sha256"].to_numpy(), y),
    }


def known_predictions(frame, maps, with_html):
    predicted = np.full(len(frame), -1, dtype=np.int8)
    exact = host = html = 0
    for idx, (u, h, digest) in enumerate(zip(frame.url, frame.host, frame.html_sha256)):
        if u in maps["url"]:
            predicted[idx] = maps["url"][u]
            exact += 1
        elif h in maps["host"]:
            predicted[idx] = maps["host"][h]
            host += 1
        elif with_html and digest and digest in maps["html_sha256"]:
            predicted[idx] = maps["html_sha256"][digest]
            html += 1
    return predicted, {"exact_url": exact, "host": host, "html_fingerprint": html}


def metrics(y, scores, known, fold_ids):
    threshold, _ = best_f1_threshold(y, scores, known)
    predicted = apply_known_labels(scores, known, threshold)
    return {
        "threshold": threshold,
        "oof_f1": float(f1_score(y, predicted)),
        "fold_f1": [float(f1_score(y[fold_ids == fold], predicted[fold_ids == fold])) for fold in sorted(set(fold_ids))],
        "known_oof_rows": int((known >= 0).sum()),
    }


def write_submission(path, scores, known, threshold):
    predictions = apply_known_labels(scores, known, threshold)
    pd.DataFrame({"Id": np.arange(len(predictions)), "Predicted": predictions}).to_csv(path, index=False)
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--train", type=Path, default=ROOT / "train_6dd2c989.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "test_5013633d.csv")
    parser.add_argument("--features", type=Path, default=ROOT / "content_features")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    experiments = ROOT / "experiments" / "url_html"
    experiments.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.features / "manifest.json").read_text(encoding="utf-8"))
    names = manifest["numeric_names"]
    raw_train, raw_test, urls, y, test_urls, hosts, test_hosts, *_ = read_inputs(args.train, args.test)
    train = load_frame(args.features / "train.jsonl.gz", names)
    test = load_frame(args.features / "test.jsonl.gz", names)
    assert train.url.tolist() == raw_train.url.fillna("").astype(str).tolist()
    assert test.url.tolist() == test_urls.tolist()
    allowed = set(urls)
    train = train[train.url.isin(allowed)].reset_index(drop=True)
    assert train.url.tolist() == urls.tolist()
    assert train.result.to_numpy().tolist() == y.tolist()
    train["host"] = hosts
    test["host"] = test_hosts
    del raw_train, raw_test
    gc.collect()

    base_oof = np.empty(len(y), dtype=np.float32)
    html_oof = np.empty(len(y), dtype=np.float32)
    base_known = np.full(len(y), -1, dtype=np.int8)
    html_known = np.full(len(y), -1, dtype=np.int8)
    fold_ids = np.empty(len(y), dtype=np.int8)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (fit_idx, valid_idx) in enumerate(cv.split(urls, y), start=1):
        fit, valid = train.iloc[fit_idx], train.iloc[valid_idx]
        print(f"Fold {fold}/{args.folds}: extracting feature matrices", flush=True)
        fitted, fit_blocks, valid_blocks = fit_features(fit, valid, names, f"fold {fold}")
        base = classifier().fit(fit_blocks["url"], y[fit_idx])
        base_oof[valid_idx] = base.decision_function(valid_blocks["url"])
        html_fit = combined(fit_blocks)
        html_valid = combined(valid_blocks)
        augmented = classifier().fit(html_fit, y[fit_idx])
        html_oof[valid_idx] = augmented.decision_function(html_valid)
        mapping = known_maps(fit, y[fit_idx])
        base_known[valid_idx], _ = known_predictions(valid, mapping, False)
        html_known[valid_idx], _ = known_predictions(valid, mapping, True)
        fold_ids[valid_idx] = fold
        print(f"Fold {fold}/{args.folds}: classifiers complete; HTML lookup covers {int((html_known[valid_idx]>=0).sum())} rows", flush=True)
        del fitted, fit_blocks, valid_blocks, base, augmented, html_fit, html_valid, fit, valid, mapping
        gc.collect()

    base_report = metrics(y, base_oof, base_known, fold_ids)
    html_report = metrics(y, html_oof, html_known, fold_ids)
    print(json.dumps({"url_baseline": base_report, "url_html": html_report}, indent=2), flush=True)
    print("Fitting final URL and URL+HTML models", flush=True)
    fitted, fit_blocks, test_blocks = fit_features(train, test, names, "full")
    base = classifier().fit(fit_blocks["url"], y)
    base_test = base.decision_function(test_blocks["url"])
    html_fit = combined(fit_blocks)
    html_test_matrix = combined(test_blocks)
    augmented = classifier().fit(html_fit, y)
    html_test = augmented.decision_function(html_test_matrix)
    mapping = known_maps(train, y)
    base_test_known, _ = known_predictions(test, mapping, False)
    html_test_known, lookup_counts = known_predictions(test, mapping, True)

    baseline_pred = write_submission(experiments / "baseline_reproduced.csv", base_test, base_test_known, base_report["threshold"])
    accepted = pd.read_csv(ROOT / "accepted" / "2026-10-05-baseline" / "submission_hybrid_svm.csv")
    reproduced_differences = int((accepted.Predicted.to_numpy() != baseline_pred).sum())
    html_path = args.out / "submission_url_html.csv"
    html_pred = write_submission(html_path, html_test, html_test_known, html_report["threshold"])
    np.savez_compressed(
        experiments / "model_scores.npz",
        y=y, fold_ids=fold_ids,
        base_oof=base_oof, html_oof=html_oof,
        base_known=base_known, html_known=html_known,
        base_test=base_test, html_test=html_test,
        base_test_known=base_test_known, html_test_known=html_test_known,
    )
    report = {
        "folds": args.folds, "seed": args.seed, "C": 1.0,
        "url_baseline": base_report, "url_html": html_report,
        "fold_f1_gain": [new - old for new, old in zip(html_report["fold_f1"], base_report["fold_f1"])],
        "overall_oof_f1_gain": html_report["oof_f1"] - base_report["oof_f1"],
        "baseline_reproduced_prediction_differences": reproduced_differences,
        "test_lookup_counts": lookup_counts,
        "test_positive_predictions": int(html_pred.sum()),
        "test_predictions_different_from_accepted_hybrid": int((html_pred != baseline_pred).sum()),
        "submission": html_path.name,
        "submission_sha256": sha256_file(html_path),
        "code_sha256": sha256_file(Path(__file__)),
        "source_archive": manifest["source"],
        "input_train_sha256": sha256_file(args.train),
        "input_test_sha256": sha256_file(args.test),
        "weights": WEIGHTS,
    }
    (experiments / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
