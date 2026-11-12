#!/usr/bin/env python3
"""Check Sphinx source files for Khmer text and valid local documentation targets."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

KHMER_RE = re.compile(r"[\u1780-\u17FF]")
THAI_RE = re.compile(r"[\u0E00-\u0E7F]")
DESCRIPTION_RE = re.compile(r"(?m)^\s*:description:\s*\S")
DOC_ROLE_RE = re.compile(r":doc:\x60([^\x60]+)\x60")
DOWNLOAD_ROLE_RE = re.compile(r":download:\x60([^\x60]+)\x60")
IMAGE_RE = re.compile(r"(?m)^\s*\.\.\s+(?:figure|image)::\s+(\S+)")
PLACEHOLDER_RE = re.compile(r"\b(?:TODO|FIXME|TBD|Lorem ipsum)\b", re.IGNORECASE)
KHMER_TYPO_RE = re.compile(r"(?:ស្វ័យប្រវតតិ|ដើម្បី្យ|បនថែម|ទំហំដ៊ីស)")


def resolve_target(source_root: Path, document: Path, target: str) -> Path:
    target = target.strip()
    if " <" in target and target.endswith(">"):
        target = target.rsplit("<", 1)[1][:-1].strip()
    target = target.split("#", 1)[0].strip()
    if target.startswith("/"):
        candidate = source_root / target.lstrip("/")
    else:
        candidate = document.parent / target
    if not candidate.suffix:
        candidate = candidate.with_suffix(".rst")
    return candidate.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("source"))
    args = parser.parse_args()

    source_root = args.source.resolve()
    if not source_root.is_dir():
        print(f"FAIL: Sphinx source directory not found: {source_root}", file=sys.stderr)
        return 1

    documents = sorted(source_root.rglob("*.rst"))
    failures: list[str] = []
    doc_links = 0
    download_links = 0
    image_links = 0

    if not documents:
        print("FAIL: no .rst files found.", file=sys.stderr)
        return 1

    for document in documents:
        try:
            content = document.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            failures.append(f"{document}: cannot read UTF-8 source: {exc}")
            continue

        relative = document.relative_to(source_root)
        if not KHMER_RE.search(content):
            failures.append(f"{relative}: no Khmer text found.")
        if THAI_RE.search(content):
            failures.append(f"{relative}: unexpected Thai-script characters found; check for mixed-language typos.")
        typo_match = KHMER_TYPO_RE.search(content)
        if typo_match:
            line_number = content.count("\n", 0, typo_match.start()) + 1
            failures.append(
                f"{relative}:{line_number}: known Khmer spelling typo {typo_match.group(0)!r} remains."
            )
        if not DESCRIPTION_RE.search(content):
            failures.append(f"{relative}: missing Sphinx :description: metadata.")

        for match in PLACEHOLDER_RE.finditer(content):
            line_number = content.count("\n", 0, match.start()) + 1
            failures.append(
                f"{relative}:{line_number}: placeholder text {match.group(0)!r} remains."
            )

        for target in DOC_ROLE_RE.findall(content):
            doc_links += 1
            resolved = resolve_target(source_root, document, target)
            if not resolved.is_file():
                failures.append(f"{relative}: unresolved :doc: target {target!r}.")

        for target in DOWNLOAD_ROLE_RE.findall(content):
            download_links += 1
            resolved = resolve_target(source_root, document, target)
            if not resolved.is_file():
                failures.append(f"{relative}: unresolved :download: target {target!r}.")

        for target in IMAGE_RE.findall(content):
            if target.startswith(("http://", "https://", "data:")):
                continue
            image_links += 1
            if target.startswith("/"):
                resolved = source_root / target.lstrip("/")
            else:
                resolved = document.parent / target
            if not resolved.is_file():
                failures.append(f"{relative}: missing image/figure file {target!r}.")

    print(
        f"Documentation-source summary: {len(documents)} RST files, "
        f"{doc_links} :doc: references, {download_links} :download: references, "
        f"{image_links} local image/figure references."
    )

    if failures:
        print(f"FAIL: {len(failures)} documentation-source issue(s):", file=sys.stderr)
        for failure in failures:
            print(f" - {failure}", file=sys.stderr)
        return 1

    print("PASS: all RST pages contain Khmer text and valid local doc/image targets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
