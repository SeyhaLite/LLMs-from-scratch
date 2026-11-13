#!/usr/bin/env python3
"""Validate the repository's Khmer Jupyter notebooks without executing them."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import nbformat

KHMER_RE = __import__("re").compile(r"[\u1780-\u17FF]")
ALLOWED_CELL_TYPES = {"markdown", "code", "raw"}


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return source if isinstance(source, str) else ""


def contains_ipython_magic(source: str) -> bool:
    """Return True when normal Python parsing is not suitable for this cell."""
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("ណូតប៊ុក-ខ្មែរ"))
    parser.add_argument("--expected-total", type=int, default=68)
    parser.add_argument("--expected-core", type=int, default=19)
    parser.add_argument("--expected-bonus", type=int, default=49)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"FAIL: notebook directory does not exist: {root}", file=sys.stderr)
        return 1

    notebooks = sorted(root.rglob("*.ipynb"))
    core = [path for path in notebooks if "បន្ថែម" not in path.relative_to(root).parts]
    bonus = [path for path in notebooks if "បន្ថែម" in path.relative_to(root).parts]

    failures: list[str] = []
    total_cells = 0
    total_code_cells = 0
    total_markdown_cells = 0
    syntax_checked = 0
    syntax_skipped = 0

    if len(notebooks) != args.expected_total:
        failures.append(
            f"Notebook count mismatch: expected {args.expected_total}, found {len(notebooks)}"
        )
    if len(core) != args.expected_core:
        failures.append(
            f"Core notebook count mismatch: expected {args.expected_core}, found {len(core)}"
        )
    if len(bonus) != args.expected_bonus:
        failures.append(
            f"Bonus notebook count mismatch: expected {args.expected_bonus}, found {len(bonus)}"
        )

    for path in notebooks:
        relative = path.relative_to(root)
        try:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook, version=4)
        except Exception as exc:
            failures.append(f"{relative}: invalid notebook JSON/schema: {exc}")
            continue

        if notebook.nbformat != 4:
            failures.append(f"{relative}: expected nbformat 4, found {notebook.nbformat}")

        cells = notebook.cells
        total_cells += len(cells)
        total_code_cells += sum(cell.cell_type == "code" for cell in cells)
        total_markdown_cells += sum(cell.cell_type == "markdown" for cell in cells)

        bad_types = sorted({cell.cell_type for cell in cells} - ALLOWED_CELL_TYPES)
        if bad_types:
            failures.append(f"{relative}: unsupported cell types: {', '.join(bad_types)}")

        all_text = "\n".join(cell_source(cell) for cell in cells)
        if not KHMER_RE.search(all_text):
            failures.append(f"{relative}: no Khmer text found in notebook cells")

        for index, cell in enumerate(cells, start=1):
            if cell.cell_type != "code":
                continue
            source = cell_source(cell)
            if not source.strip() or contains_ipython_magic(source):
                syntax_skipped += 1
                continue
            try:
                ast.parse(source, filename=f"{relative}:cell-{index}")
                syntax_checked += 1
            except SyntaxError as exc:
                failures.append(
                    f"{relative}: Python syntax error in cell {index}, "
                    f"line {exc.lineno}: {exc.msg}"
                )

    print(
        "Notebook validation summary: "
        f"{len(notebooks)} files ({len(core)} core, {len(bonus)} bonus), "
        f"{total_cells} cells ({total_code_cells} code, "
        f"{total_markdown_cells} Markdown), "
        f"{syntax_checked} Python cells syntax-checked, "
        f"{syntax_skipped} cells skipped (empty/IPython magic)."
    )

    if failures:
        print(f"FAIL: {len(failures)} validation issue(s):", file=sys.stderr)
        for failure in failures:
            print(f" - {failure}", file=sys.stderr)
        return 1

    print("PASS: all notebooks passed JSON, nbformat schema, cell and Khmer checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
