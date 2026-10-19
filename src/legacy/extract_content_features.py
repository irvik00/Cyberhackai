"""Derive compact HTML features from the organizer ZIP in a streaming pass."""

import csv
import gzip
import hashlib
import html
import io
import json
import math
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import urljoin, urlsplit

ROOT = Path(__file__).resolve().parent
MAX_HTML = 600_000
MAX_TEXT = 8_000
MAX_TAGS = 1_500
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
csv.field_size_limit(1 << 30)

TITLE = re.compile(r"<title\b[^>]*>(.*?)</title\s*>", re.I | re.S)
SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.I | re.S)
COMMENTS = re.compile(r"<!--.*?-->", re.S)
TAGS = re.compile(r"<\s*([a-z][a-z0-9]*)\b", re.I)
STRIP_TAGS = re.compile(r"<[^>]+>", re.S)
SPACE = re.compile(r"\s+")
ATTR = re.compile(r"""\b(?:href|src|action)\s*=\s*["']([^"']+)["']""", re.I)
FORM = re.compile(r"<form\b([^>]*)>", re.I | re.S)
ACTION = re.compile(r"""\baction\s*=\s*["']([^"']*)["']""", re.I)
INPUT = re.compile(r"<input\b([^>]*)>", re.I | re.S)
TYPE = re.compile(r"""\btype\s*=\s*["']?([a-z]+)""", re.I)
WORDS = re.compile(r"[a-z]{4,}")
BRANDS = ("facebook", "paypal", "microsoft", "amazon", "apple", "google", "outlook",
          "office", "docusign", "linkedin", "whatsapp", "yahoo", "netflix", "dropbox")
NUMERIC_NAMES = [
    "log_html_length", "log_visible_length", "log_title_length", "html_truncated",
    "log_tags", "log_forms", "log_inputs", "log_passwords", "log_hidden_inputs",
    "log_email_inputs", "log_post_forms", "log_external_forms", "log_empty_actions",
    "log_anchors", "log_images", "log_scripts", "log_styles", "log_iframes",
    "log_textareas", "log_selects", "log_buttons", "log_metas", "log_resources",
    "log_external_resources", "log_unique_external_hosts", "external_resource_ratio",
    "log_http_resources_on_https", "log_meta_refresh", "title_brand_count",
    "title_brand_host_mismatch", "title_host_word_overlap", "no_content", "no_title",
]


def hostname(url):
    try:
        candidate = url if "://" in url else "//" + url.lstrip("/")
        return (urlsplit(candidate).hostname or "").lower()
    except ValueError:
        return ""


def normalized_host(url):
    return hostname(url).removeprefix("www.")


def absolute_target(base_url, reference):
    try:
        return urljoin(base_url, html.unescape(reference))
    except ValueError:
        return ""


def visible(value):
    return SPACE.sub(" ", html.unescape(STRIP_TAGS.sub(" ", value))).strip()


def extract(url, source):
    raw = source or ""
    if raw.strip().lower() in ("none", "null", "nan"):
        raw = ""
    body = raw[:MAX_HTML]
    title_match = TITLE.search(body)
    title = visible(title_match.group(1))[:1000] if title_match else ""
    without_scripts = COMMENTS.sub(" ", SCRIPT_STYLE.sub(" ", body))
    text = visible(without_scripts)
    tag_list = [item.lower() for item in TAGS.findall(body)]
    tags = Counter(tag_list)
    main_host = normalized_host(url)
    refs = ATTR.findall(body)[:2000]
    resource_hosts = []
    external_hosts = []
    insecure_resources = 0
    for ref in refs:
        if ref.startswith(("#", "data:", "mailto:", "javascript:", "tel:")):
            continue
        target = absolute_target(url, ref)
        host = normalized_host(target)
        if host:
            resource_hosts.append(host)
            if host != main_host:
                external_hosts.append(host)
            insecure_resources += int(url.lower().startswith("https://") and target.lower().startswith("http://"))

    form_attrs = FORM.findall(body)
    external_forms = empty_actions = post_forms = 0
    for attrs in form_attrs:
        post_forms += int(bool(re.search(r"""\bmethod\s*=\s*["']?post\b""", attrs, re.I)))
        action_match = ACTION.search(attrs)
        if not action_match or not action_match.group(1).strip():
            empty_actions += 1
        else:
            target_host = normalized_host(absolute_target(url, action_match.group(1)))
            external_forms += int(bool(target_host and target_host != main_host))
    input_types = Counter()
    for attrs in INPUT.findall(body):
        match = TYPE.search(attrs)
        input_types[match.group(1).lower() if match else "text"] += 1
    lower_title = title.lower()
    title_brands = [brand for brand in BRANDS if brand in lower_title]
    title_words = WORDS.findall(lower_title)
    meta_refresh = len(re.findall(r"<meta\b[^>]*http-equiv\s*=\s*[\"']?refresh", body, re.I))
    numeric = [
        math.log1p(len(raw)), math.log1p(len(text)), math.log1p(len(title)),
        int(len(raw) > MAX_HTML), math.log1p(len(tag_list)), math.log1p(tags["form"]),
        math.log1p(tags["input"]), math.log1p(input_types["password"]),
        math.log1p(input_types["hidden"]), math.log1p(input_types["email"]),
        math.log1p(post_forms), math.log1p(external_forms), math.log1p(empty_actions),
        math.log1p(tags["a"]), math.log1p(tags["img"]), math.log1p(tags["script"]),
        math.log1p(tags["style"]), math.log1p(tags["iframe"]), math.log1p(tags["textarea"]),
        math.log1p(tags["select"]), math.log1p(tags["button"]), math.log1p(tags["meta"]),
        math.log1p(len(resource_hosts)), math.log1p(len(external_hosts)),
        math.log1p(len(set(external_hosts))), len(external_hosts) / max(len(resource_hosts), 1),
        math.log1p(insecure_resources), math.log1p(meta_refresh), len(title_brands),
        sum(brand not in main_host for brand in title_brands),
        sum(word in main_host for word in title_words) / max(len(title_words), 1),
        int(not raw), int(not title),
    ]
    assert len(numeric) == len(NUMERIC_NAMES)
    return {
        "title": title,
        "text": text[:MAX_TEXT],
        "link_hosts": " ".join(resource_hosts[:300]),
        "tags": " ".join(tag_list[:MAX_TAGS]),
        "numeric": numeric,
        "html_sha256": hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest() if len(raw) >= 1000 else "",
    }


def process_member(archive, member, reference_csv, output_path, expected_rows):
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    count = available = titles = 0
    hashes = set()
    with reference_csv.open("r", encoding="utf-8-sig", newline="") as base_file:
        base_rows = csv.DictReader(base_file)
        with archive.open(member) as zipped, io.TextIOWrapper(zipped, encoding="utf-8-sig", errors="replace", newline="") as text_file:
            source_rows = csv.DictReader(text_file)
            if "url" not in source_rows.fieldnames or "content" not in source_rows.fieldnames:
                raise ValueError(f"Unexpected columns in {member}: {source_rows.fieldnames}")
            with temporary.open("wb") as destination:
                with gzip.GzipFile(filename="", fileobj=destination, mode="wb", mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as output:
                        for row in source_rows:
                            expected = next(base_rows, None)
                            if expected is None or expected["url"] != row["url"]:
                                raise ValueError(f"URL/order mismatch in {member} at row {count}")
                            if "result" in expected and expected["result"] != row["result"]:
                                raise ValueError(f"Training label mismatch at row {count}")
                            record = {"url": expected["url"], **extract(expected["url"], row.get("content", ""))}
                            if "result" in expected:
                                record["result"] = int(expected["result"])
                            available += int(not record["numeric"][-2])
                            titles += int(bool(record["title"]))
                            if record["html_sha256"]:
                                hashes.add(record["html_sha256"])
                            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                            count += 1
                            if count % 2000 == 0:
                                print(f"{member}: {count}/{expected_rows} pages processed", flush=True)
        if next(base_rows, None) is not None or count != expected_rows:
            raise ValueError(f"Unexpected row count for {member}: {count}")
    temporary.replace(output_path)
    return {"rows": count, "available_content": available, "pages_with_title": titles, "distinct_nonempty_html_hashes": len(hashes)}


def main():
    folder = ROOT / "content_features"
    folder.mkdir(parents=True, exist_ok=True)
    records = {}
    with zipfile.ZipFile(ROOT / "data" / "organizer_pages.zip") as archive:
        for name, reference, expected in [
            ("train", "train_6dd2c989.csv", 64_000),
            ("test", "test_5013633d.csv", 16_000),
        ]:
            records[name] = process_member(archive, f"{name}_with_content.csv", ROOT / reference, folder / f"{name}.jsonl.gz", expected)
    schema = {
        "numeric_names": NUMERIC_NAMES,
        "max_html_characters": MAX_HTML,
        "max_visible_text_characters": MAX_TEXT,
        "max_tag_tokens": MAX_TAGS,
        "source": json.loads((ROOT / "data" / "organizer_source.json").read_text(encoding="utf-8")),
        "datasets": records,
    }
    (folder / "manifest.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(schema, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
