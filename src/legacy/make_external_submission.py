"""Create a reproducible classifier with cautious public external URL evidence."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from audit_external_sources import canonical_url
from refine_predictions import evaluate
from train_model import apply_known_labels, read_inputs

ROOT = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE_THRESHOLD = 0.07130044297070283
DISTANCE = 0.25


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-weight", type=float, default=0.0)
    args = parser.parse_args()
    if not 0 <= args.source_weight <= 1:
        raise ValueError("source-weight must be between 0 and 1")
    folder = ROOT / "experiments" / "external"
    _, _, urls, y, test_urls, *_ = read_inputs(ROOT / "train_6dd2c989.csv", ROOT / "test_5013633d.csv")
    previous = json.loads((ROOT / "experiments/refinements/report.json").read_text(encoding="utf-8"))
    for name, expected in previous["input_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError("Input data changed since the scores were generated")
    saved = np.load(ROOT / "experiments/refinements/refined_scores.npz")
    assert np.array_equal(y, saved["y"])
    scores, test_scores = saved["oof"].copy(), saved["test"].copy()
    scales = {}
    if args.source_weight:
        source = np.load(ROOT / "experiments/source_refinement/model_scores.npz")
        assert np.array_equal(y, source["y"]) and np.array_equal(saved["fold_ids"], source["fold_ids"])
        scales = {"refined": float(scores.std()), "source": float(source["oof"].std())}
        scores = (1 - args.source_weight) * scores / scales["refined"] + args.source_weight * source["oof"] / scales["source"]
        test_scores = (1 - args.source_weight) * test_scores / scales["refined"] + args.source_weight * source["test"] / scales["source"]
    known, test_known = saved["known"].copy(), saved["test_known"].copy()
    rules = np.load(folder / "host_rules.npz")
    rule, test_rule = rules["train_support_1"], rules["test_support_1"]
    use = (known < 0) & (rule >= 0) & (np.abs(saved["oof"] - BASE_THRESHOLD) <= DISTANCE)
    known[use] = rule[use]
    test_host_use = (test_known < 0) & (test_rule >= 0) & (np.abs(saved["test"] - BASE_THRESHOLD) <= DISTANCE)
    test_known[test_host_use] = test_rule[test_host_use]
    public_labels = json.loads((folder / "matching_labels.json").read_text(encoding="utf-8"))
    pure = {key: next(iter(set(sources.values()))) for key, sources in public_labels.items()
            if len(set(sources.values())) == 1}
    exact_count = 0
    for idx, url in enumerate(test_urls):
        key = canonical_url(url)
        if saved["test_known"][idx] < 0 and abs(saved["test"][idx] - BASE_THRESHOLD) <= DISTANCE and key in pure:
            test_known[idx] = pure[key]
            exact_count += 1
    metrics = evaluate(y, scores, known, saved["fold_ids"])
    baseline = evaluate(y, saved["oof"], saved["known"], saved["fold_ids"])
    predictions = apply_known_labels(test_scores, test_known, metrics["threshold"])
    sample = pd.read_csv(ROOT / "sample_460901b3.csv")
    old = pd.read_csv(ROOT / "submission_url_html.csv")
    refined = pd.read_csv(ROOT / "submission_html_refined.csv")
    assert len(predictions) == len(test_urls) == len(sample) == 16000
    assert sample.Id.tolist() == list(range(16000))
    assert set(np.unique(predictions)) <= {0, 1}
    output = ROOT / "submission_external.csv"
    pd.DataFrame({"Id": np.arange(16000), "Predicted": predictions}).to_csv(output, index=False)
    source_names = ["make_external_submission.py", "audit_external_sources.py", "external_host_refinement.py",
                    "download_external_data.py", "refine_predictions.py"]
    if args.source_weight:
        source_names += ["train_source_refinement.py", "extract_source_features.py"]
    report = {
        "parameters": {"external_score_distance": DISTANCE, "external_minimum_unique_host_urls": 1,
                       "refined_threshold_for_confidence": BASE_THRESHOLD,
                       "source_weight": args.source_weight, "scales": scales},
        "baseline_refined": baseline, "cv_with_decontaminated_external_hosts": metrics,
        "cv_protocol": "Every local training URL was removed from external data before building host evidence. External exact-overlap labels are used only at test prediction time, and are not included in the CV score.",
        "test_external_host_rule_rows": int(test_host_use.sum()),
        "test_external_exact_rule_rows": exact_count,
        "test_predictions_different_from_accepted_html": int((predictions != old.Predicted.to_numpy()).sum()),
        "test_predictions_different_from_accepted_refined": int((predictions != refined.Predicted.to_numpy()).sum()),
        "test_rows": 16000, "test_positive_predictions": int(predictions.sum()),
        "submission": output.name, "submission_sha256": sha256_file(output),
        "source_sha256": {name: sha256_file(ROOT / name) for name in source_names},
        "external_sources": json.loads((ROOT / "data/external/sources.json").read_text(encoding="utf-8")),
        "public_f1": None,
    }
    (folder / "submission_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
