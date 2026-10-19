"""Train the external solution from cached HTML features."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    required = ["content_features/train.jsonl.gz", "content_features/test.jsonl.gz", "source_features/train.jsonl.gz", "source_features/test.jsonl.gz",
                "data/external/malicious_urls_v1.zip", "data/external/url_dataset_v1.zip", "data/external/phiusiil_uci_967.zip"]
    for name in required:
        assert (ROOT / name).exists(), name
    for script in ["train_content_model.py", "train_numeric_refinement.py", "make_refined_submission.py", "train_source_refinement.py",
                   "audit_external_sources.py", "external_host_refinement.py", "replay_external_frozen.py"]:
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, "-X", "utf8", str(ROOT / script)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
