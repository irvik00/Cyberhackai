"""Combine validated SVM, numeric boosting and cautious HTML template lookup."""

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from refine_predictions import evaluate, fingerprint, label_map, lookup, oof_lookup
from train_model import apply_known_labels, read_inputs

ROOT = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
NUMERIC_WEIGHT = 0.25
TEMPLATE_SCORE_DISTANCE = 0.5
BASE_THRESHOLD = 0.034618228673934937


def template_keys(name, wanted):
    urls, keys = [], []
    with gzip.open(ROOT / "content_features" / f"{name}.jsonl.gz", "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["url"] in wanted:
                urls.append(row["url"])
                tags = row["tags"]
                keys.append(fingerprint(tags) if len(tags.split()) >= 30 else "")
    return urls, np.asarray(keys, dtype=object)


def cautious_known(base, rule, svm_scores):
    result = base.copy()
    allowed = (base < 0) & (rule >= 0) & (np.abs(svm_scores - BASE_THRESHOLD) <= TEMPLATE_SCORE_DISTANCE)
    result[allowed] = rule[allowed]
    return result


def logit(probabilities):
    clipped = np.clip(probabilities.astype(float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def sha256_file(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def main():
    output = ROOT / "experiments" / "refinements"
    output.mkdir(parents=True, exist_ok=True)
    _, _, urls, y, test_urls, *_ = read_inputs(ROOT / "train_6dd2c989.csv", ROOT / "test_5013633d.csv")
    svm = np.load(ROOT / "experiments/url_html/model_scores.npz")
    numeric = np.load(ROOT / "experiments/numeric_refinement/model_scores.npz")
    assert np.array_equal(y, svm["y"]) and np.array_equal(y, numeric["y"])
    assert np.array_equal(svm["fold_ids"], numeric["fold_ids"])
    order, keys = template_keys("train", set(urls))
    test_order, test_keys = template_keys("test", set(test_urls))
    assert order == urls.tolist() and test_order == test_urls.tolist()
    folds = svm["fold_ids"]
    template_oof = oof_lookup(keys, y, folds, minimum=1)
    template_test = lookup(test_keys, label_map(keys, y, minimum=1))
    known = cautious_known(svm["html_known"], template_oof, svm["html_oof"])
    test_known = cautious_known(svm["html_test_known"], template_test, svm["html_test"])
    html_scale = float(svm["html_oof"].std())
    numeric_scale = float(logit(numeric["oof"]).std())
    combined_oof = (1 - NUMERIC_WEIGHT) * svm["html_oof"] / html_scale + NUMERIC_WEIGHT * logit(numeric["oof"]) / numeric_scale
    combined_test = (1 - NUMERIC_WEIGHT) * svm["html_test"] / html_scale + NUMERIC_WEIGHT * logit(numeric["test"]) / numeric_scale
    metrics = evaluate(y, combined_oof, known, folds)
    baseline = evaluate(y, svm["html_oof"], svm["html_known"], folds)
    pred = apply_known_labels(combined_test, test_known, metrics["threshold"])
    old = pd.read_csv(ROOT / "submission_url_html.csv")
    sample = pd.read_csv(ROOT / "sample_460901b3.csv")
    assert len(pred) == len(test_urls) == len(sample) == 16000
    assert sample.Id.tolist() == list(range(len(pred)))
    assert set(np.unique(pred)) <= {0, 1}
    target = ROOT / "submission_html_refined.csv"
    pd.DataFrame({"Id": np.arange(len(pred)), "Predicted": pred}).to_csv(target, index=False)
    comparison = {
        "parameters": {"numeric_weight": NUMERIC_WEIGHT, "template_score_distance": TEMPLATE_SCORE_DISTANCE,
                       "base_svm_threshold": BASE_THRESHOLD, "html_scale": html_scale, "numeric_logit_scale": numeric_scale},
        "baseline": baseline, "refined": metrics,
        "fold_f1_gain": [new - previous for new, previous in zip(metrics["fold_f1"], baseline["fold_f1"])],
        "test_rows": len(pred), "test_positive_predictions": int(pred.sum()),
        "test_predictions_different_from_accepted_html": int((pred != old.Predicted.to_numpy()).sum()),
        "test_template_lookup_rows": int(((test_known >= 0) & (svm["html_test_known"] < 0)).sum()),
        "submission": target.name, "submission_sha256": sha256_file(target),
        "source_sha256": {name: sha256_file(ROOT / name) for name in (
            "make_refined_submission.py", "refine_predictions.py", "train_numeric_refinement.py",
            "train_model.py", "train_content_model.py", "extract_content_features.py")},
        "input_sha256": {name: sha256_file(ROOT / name) for name in ("train_6dd2c989.csv", "test_5013633d.csv")},
        "public_f1": None,
    }
    (output / "report.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    np.savez_compressed(output / "refined_scores.npz", y=y, fold_ids=folds, oof=combined_oof,
                        test=combined_test, known=known, test_known=test_known)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
