"""Reproduce submission_external.csv."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def config() -> dict:
    return json.loads((ROOT / "configs/external.json").read_text(encoding="utf-8"))


def checked(path: Path, expected: str) -> None:
    if not path.is_file() or sha(path) != expected:
        raise ValueError(f"Missing or changed file: {path}")


def replay(output: Path) -> None:
    import numpy as np
    c = config()
    provenance = json.loads((ROOT / "artifacts/provenance.json").read_text(encoding="utf-8"))
    artifact = ROOT / "artifacts/external_test_scores.npz"
    checked(artifact, provenance["test_artifact_sha256"])
    with np.load(artifact, allow_pickle=False) as saved:
        refined, source, known = saved["refined_test"], saved["source_test"], saved["known_test"]
        score = (1-c["source_weight"]) * refined / c["refined_scale"] + c["source_weight"] * source / c["source_scale"]
    if not (len(score) == len(known) == c["test_rows"] and np.isfinite(score).all()):
        raise ValueError("Invalid saved test arrays")
    if not set(np.unique(known)) <= {-1, 0, 1}:
        raise ValueError("Invalid known-rule decisions")
    predicted = (score >= c["threshold"]).astype(np.int64)
    predicted[known >= 0] = known[known >= 0]
    output.parent.mkdir(parents=True, exist_ok=True)
    reference = ROOT / "reference/submission_external.csv"
    checked(reference, c["submission_sha256"])
    if output.resolve() == reference.resolve():
        raise ValueError("Choose an output path outside reference/")
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\r\n")
        writer.writerow(["Id", "Predicted"])
        writer.writerows(enumerate(map(int, predicted)))
    with reference.open(encoding="utf-8-sig", newline="") as stream:
        expected = list(csv.DictReader(stream))
    differences = sum(int(row["Predicted"]) != int(predicted[i]) for i, row in enumerate(expected))
    result = {"mode": "replay_saved_scores", "classifier_training": False, "rows": len(predicted),
        "differences_from_submitted": differences, "sha256": sha(output), "expected_sha256": c["submission_sha256"],
        "byte_identical": sha(output) == c["submission_sha256"], "private_f1_recalculated": False}
    output.with_suffix(".report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if differences or not result["byte_identical"]:
        raise RuntimeError("The reproduced CSV differs from the accepted submission")


def install_code(work: Path) -> None:
    provenance = json.loads((ROOT / "artifacts/provenance.json").read_text(encoding="utf-8"))
    for name, expected in provenance["runtime_code_sha256"].items():
        source = ROOT / name
        checked(source, expected)
        shutil.copy2(source, work / source.name)
    for name in ["submission_external.csv", "submission_hybrid_svm.csv", "submission_url_html.csv", "submission_html_refined.csv"]:
        shutil.copy2(ROOT / "reference" / name, work / name)
    baseline = work / "accepted/2026-10-05-baseline/submission_hybrid_svm.csv"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "reference/submission_hybrid_svm.csv", baseline)
    target = work / "experiments/external/submission_report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "reference/external_submission_report.json", target)
    data = work / "data"
    data.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "reference/organizer_source.json", data / "organizer_source.json")


def from_bundle(bundle: Path, work: Path) -> None:
    c = config()
    checked(bundle, c["verification_bundle_sha256"])
    with zipfile.ZipFile(bundle) as archive:
        records = json.loads(archive.read("verification_manifest.json"))["files"]
        names = ["train_6dd2c989.csv", "test_5013633d.csv", "sample_460901b3.csv",
            "content_features/train.jsonl.gz", "content_features/test.jsonl.gz", "content_features/manifest.json",
            "source_features/train.jsonl.gz", "source_features/test.jsonl.gz", "source_features/manifest.json",
            "data/external/malicious_urls_v1.zip", "data/external/url_dataset_v1.zip", "data/external/phiusiil_uci_967.zip", "data/external/sources.json"]
        for name in names:
            destination = work / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=8*1024*1024)
            checked(destination, records[name]["sha256"])
    print("Training inputs prepared.", flush=True)


def run(work: Path, script: str) -> None:
    subprocess.run([sys.executable, "-X", "utf8", str(work / script)], cwd=work, check=True)


def from_original_inputs(args, work: Path) -> None:
    c = config()
    for name, aliases in [("train_6dd2c989.csv", ["train_6dd2c989.csv", "train.csv"]),
                          ("test_5013633d.csv", ["test_5013633d.csv", "test.csv"]),
                          ("sample_460901b3.csv", ["sample_460901b3.csv", "sample.csv"])]:
        path = next((args.competition_dir / a for a in aliases if (args.competition_dir / a).is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Put {name} into {args.competition_dir}")
        checked(path, c["competition_sha256"][name])
        shutil.copy2(path, work / name)
    organizer = work / "data/organizer_pages.zip"
    if args.organizer_archive:
        checked(args.organizer_archive, c["organizer_archive_sha256"])
        shutil.copy2(args.organizer_archive, organizer)
    else:
        print("Downloading the organizer dataset.", flush=True)
        run(work, "download_organizer_archive.py")
    checked(organizer, c["organizer_archive_sha256"])
    if args.external_dir:
        folder = work / "data/external"
        folder.mkdir(exist_ok=True)
        for item in json.loads((ROOT / "reference/external_sources.json").read_text(encoding="utf-8")):
            path = args.external_dir / item["file"]
            checked(path, item["sha256"])
            shutil.copy2(path, folder / item["file"])
    # Checks pinned versions if already copied; otherwise downloads them.
    run(work, "download_external_data.py")
    run(work, "extract_content_features.py")
    run(work, "extract_source_features.py")


def train(args) -> None:
    c = config()
    work = args.workdir.resolve()
    if work.exists():
        raise FileExistsError(f"Use a new --workdir; existing files are preserved: {work}")
    work.mkdir(parents=True)
    install_code(work)
    if args.bundle:
        from_bundle(args.bundle.resolve(), work)
    else:
        from_original_inputs(args, work)
    run(work, "run_full_training.py")
    generated = work / "reproduced_submission_external.csv"
    checked(generated, c["submission_sha256"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generated, args.output)
    report = json.loads((work / "verification_replay_report.json").read_text(encoding="utf-8"))
    report.update({"mode": "full_training_from_features", "classifier_training": True, "workspace": str(work)})
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Verified full-training output: {args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    quick = actions.add_parser("replay", help="Rebuild the exact CSV from included scores; no training")
    quick.add_argument("--output", type=Path, default=ROOT / "outputs/submission_external.csv")
    full = actions.add_parser("train", help="Train the original classifiers and verify the exact CSV")
    source = full.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path, help="solution_external_verification.zip")
    source.add_argument("--competition-dir", type=Path, help="Directory with the three original competition CSV files")
    full.add_argument("--organizer-archive", type=Path, help="Optional local original HTML ZIP; otherwise download it")
    full.add_argument("--external-dir", type=Path, help="Optional local three pinned public external ZIP files")
    full.add_argument("--workdir", type=Path, default=ROOT / "_work/external")
    full.add_argument("--output", type=Path, default=ROOT / "outputs/submission_external.csv")
    args = parser.parse_args()
    if args.action == "replay":
        replay(args.output)
    else:
        train(args)


if __name__ == "__main__":
    main()
