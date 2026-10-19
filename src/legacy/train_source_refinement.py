"""Compare a source-code/HTML-attribute SVM with the accepted refined model."""

import gc
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

from refine_predictions import evaluate
from train_model import apply_known_labels, best_f1_threshold, read_inputs

ROOT = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(name, wanted):
    with gzip.open(ROOT / "source_features" / f"{name}.jsonl.gz", "rt", encoding="utf-8") as source:
        frame = pd.DataFrame([json.loads(line) for line in source])
    return frame[frame.url.isin(wanted)].reset_index(drop=True)


def fit_features(fit, valid):
    fit_blocks, valid_blocks = [], []
    for name, maximum, weight in [("source", 250000, 1.0), ("attributes", 150000, 1.0)]:
        transformer = TfidfVectorizer(token_pattern=r"(?u)[\w.-]{2,}", ngram_range=(1, 2),
            min_df=2, max_features=maximum, sublinear_tf=True, dtype=np.float32)
        fit_blocks.append(transformer.fit_transform(fit[name]) * weight)
        valid_blocks.append(transformer.transform(valid[name]) * weight)
        print(f"Source branch {name}: {fit_blocks[-1].shape[1]} features", flush=True)
    return hstack(fit_blocks, format="csr"), hstack(valid_blocks, format="csr")


def model():
    return LinearSVC(C=1.0, class_weight="balanced", max_iter=10000, dual="auto", random_state=42)


def comparison(y, base, new, known, folds):
    s_base, s_new = float(base.std()), float(new.std())
    rows = []
    for weight in (0, 0.1, 0.25, 0.5, 0.75, 1.0):
        scores = (1 - weight) * base / s_base + weight * new / s_new
        rows.append({"source_weight": weight, **evaluate(y, scores, known, folds)})
    return {"scales": {"base": s_base, "source": s_new}, "candidates": rows}


def main():
    output = ROOT / "experiments/source_refinement"
    output.mkdir(parents=True, exist_ok=True)
    _, _, urls, y, test_urls, *_ = read_inputs(ROOT / "train_6dd2c989.csv", ROOT / "test_5013633d.csv")
    train = load("train", set(urls))
    assert train.url.tolist() == urls.tolist()
    saved = np.load(ROOT / "experiments/refinements/refined_scores.npz")
    assert np.array_equal(y, saved["y"])
    folds, scores = saved["fold_ids"], np.empty(len(y), dtype=np.float32)
    for fold in range(1, 6):
        fit, valid = folds != fold, folds == fold
        print(f"Source model fold {fold}/5", flush=True)
        x_fit, x_valid = fit_features(train.loc[fit], train.loc[valid])
        classifier = model().fit(x_fit, y[fit])
        scores[valid] = classifier.decision_function(x_valid)
        del x_fit, x_valid, classifier
        gc.collect()
        if fold == 2:
            selected = folds <= 2
            base, source, known = saved["oof"][selected], scores[selected], saved["known"][selected]
            yy = y[selected]
            old_threshold, old_f1 = best_f1_threshold(yy, base, known)
            best = old_f1
            for w in (0.1, 0.25, 0.5, 0.75, 1.0):
                blended = (1 - w) * base / base.std() + w * source / source.std()
                threshold, score = best_f1_threshold(yy, blended, known)
                best = max(best, score)
            print(json.dumps({"pilot_baseline_f1": old_f1, "pilot_best_blend_f1": best}), flush=True)
            if best - old_f1 < 0.0001:
                (output / "report.json").write_text(json.dumps({"status": "stopped_after_two_folds", "pilot_baseline_f1": old_f1, "pilot_best_blend_f1": best}, indent=2), encoding="utf-8")
                print("No meaningful improvement in the first two folds; experiment stopped", flush=True)
                return
    report = comparison(y, saved["oof"], scores, saved["known"], folds)
    print(json.dumps(report, indent=2), flush=True)
    test = load("test", set(test_urls))
    assert test.url.tolist() == test_urls.tolist()
    x_fit, x_test = fit_features(train, test)
    final = model().fit(x_fit, y)
    test_scores = final.decision_function(x_test)
    np.savez_compressed(output / "model_scores.npz", y=y, fold_ids=folds, oof=scores, test=test_scores)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
