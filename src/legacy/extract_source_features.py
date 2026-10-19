"""Extract inert HTML source snippets and form/resource attributes."""

import csv
import gzip
import html
import io
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
csv.field_size_limit(1 << 30)
ATTRIBUTE = re.compile(r'''\b(id|name|class|action|href|src|type|method|placeholder|autocomplete|rel|content|data-[\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))''', re.I)
TOKENS = re.compile(r"(?u)[\w.-]{2,}")


def features(raw):
    raw = raw or ""
    body = raw[:200_000]
    attrs = []
    for match in ATTRIBUTE.finditer(body):
        name = match.group(1).lower()
        value = html.unescape(next(value for value in match.groups()[1:] if value is not None))[:500]
        tokens = TOKENS.findall(value.lower())[:50]
        attrs.extend(name + "_" + token for token in tokens)
        if len(attrs) >= 1800:
            break
    snippet = raw[:12_000] + ("\n" + raw[-4000:] if len(raw) > 12_000 else "")
    snippet = re.sub(r"[a-fA-F0-9]{40,}", " LONGHEX ", snippet)
    snippet = re.sub(r"[A-Za-z0-9+/=]{160,}", " LONGDATA ", snippet)
    return {"source": snippet, "attributes": " ".join(attrs)[:16_000]}


def main():
    output = ROOT / "source_features"
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    with zipfile.ZipFile(ROOT / "data/organizer_pages.zip") as archive:
        for name, csv_name, expected_count in [("train", "train_6dd2c989.csv", 64000), ("test", "test_5013633d.csv", 16000)]:
            path = output / f"{name}.jsonl.gz"
            temp = path.with_suffix(path.suffix + ".part")
            count = 0
            with (ROOT / csv_name).open(encoding="utf-8-sig", newline="") as reference:
                original = csv.DictReader(reference)
                with archive.open(f"{name}_with_content.csv") as binary, io.TextIOWrapper(binary, encoding="utf-8-sig", errors="replace", newline="") as text:
                    with temp.open("wb") as target, gzip.GzipFile(filename="", fileobj=target, mode="wb", mtime=0) as zipped, io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as result:
                        for row in csv.DictReader(text):
                            correct = next(original, None)
                            if correct is None or correct["url"] != row["url"]:
                                raise ValueError(f"Unexpected URL/order at {name} row {count}")
                            if "result" in correct and correct["result"] != row["result"]:
                                raise ValueError(f"Unexpected label at train row {count}")
                            record = {"url": correct["url"], **features(row.get("content", ""))}
                            result.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                            count += 1
                            if count % 4000 == 0:
                                print(f"Source features {name}: {count}/{expected_count}", flush=True)
                if next(original, None) is not None or count != expected_count:
                    raise ValueError(f"Unexpected row count: {count}")
            temp.replace(path)
            counts[name] = count
    (output / "manifest.json").write_text(json.dumps({"rows": counts, "attribute_html_limit": 200000,
        "source_head_characters": 12000, "source_tail_characters": 4000,
        "maximum_attribute_characters": 16000,
        "source": json.loads((ROOT / "data/organizer_source.json").read_text(encoding="utf-8"))}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
