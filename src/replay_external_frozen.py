"""Rebuild submission_external.csv from saved branch scores."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from audit_external_sources import canonical_url

EXPECTED_SHA = "17301496020fb14600e2c75b1b9d4caaa6843c21cdd06fb32678a4eb1b0499f9"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    original = ROOT / "submission_external.csv"
    assert digest(original) == EXPECTED_SHA, "Original submission differs from the accepted file"
    report = json.loads((ROOT / "experiments/external/submission_report.json").read_text(encoding="utf-8"))
    params = report["parameters"]
    threshold = report["cv_with_decontaminated_external_hosts"]["threshold"]
    assert params["source_weight"] == 0.25 and threshold == 0.034253455073786174
    with np.load(ROOT / "experiments/refinements/refined_scores.npz") as ref, np.load(ROOT / "experiments/source_refinement/model_scores.npz") as source:
        assert np.array_equal(ref["y"], source["y"]) and np.array_equal(ref["fold_ids"], source["fold_ids"])
        refined_test = ref["test"].copy()
        original_known = ref["test_known"].copy()
        known = original_known.copy()
        scores = (1-params["source_weight"]) * refined_test / params["scales"]["refined"] + params["source_weight"] * source["test"] / params["scales"]["source"]
    with np.load(ROOT / "experiments/external/host_rules.npz") as rules:
        host_rule = rules["test_support_1"]
    active = (known < 0) & (host_rule >= 0) & (np.abs(refined_test-params["refined_threshold_for_confidence"]) <= params["external_score_distance"])
    known[active] = host_rule[active]
    external = json.loads((ROOT / "experiments/external/matching_labels.json").read_text(encoding="utf-8"))
    pure = {key: next(iter(set(values.values()))) for key, values in external.items() if len(set(values.values())) == 1}
    with (ROOT / "test_5013633d.csv").open(encoding="utf-8-sig", newline="") as stream:
        urls = [row["url"] for row in csv.DictReader(stream)]
    for i, url in enumerate(urls):
        key = canonical_url(url)
        if original_known[i] < 0 and abs(refined_test[i]-params["refined_threshold_for_confidence"]) <= params["external_score_distance"] and key in pure:
            known[i] = pure[key]
    predicted = (scores >= threshold).astype(np.int64)
    predicted[known >= 0] = known[known >= 0]
    with original.open(encoding="utf-8-sig", newline="") as stream:
        expected = list(csv.DictReader(stream))
    assert len(urls) == len(predicted) == len(expected) == 16000 and np.isfinite(scores).all()
    assert all(row["Id"] == str(i) for i, row in enumerate(expected))
    differences = int(np.sum(predicted != np.asarray([int(row["Predicted"]) for row in expected])))
    output = ROOT / "reproduced_submission_external.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\r\n")
        writer.writerow(["Id", "Predicted"])
        writer.writerows((i, int(value)) for i, value in enumerate(predicted))
    result = {"mode": "replay_saved_scores_no_classifier_training", "rows": len(predicted), "differences_from_submitted": differences,
        "threshold": threshold, "source_weight": params["source_weight"], "sha256": digest(output), "expected_sha256": EXPECTED_SHA,
        "file_identical": digest(output) == EXPECTED_SHA, "original_submission_preserved": digest(original) == EXPECTED_SHA,
        "public_f1": 0.995079, "private_f1": 0.992305,
        "private_labels_available": False, "private_f1_recalculated": False}
    (ROOT / "verification_replay_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert differences == 0 and result["file_identical"], "Reproduction differs; inspect the saved report"


if __name__ == "__main__":
    main()
