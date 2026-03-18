#!/usr/bin/env python3
"""
pdf_to_markdown.py
------------------
Convert a PDF file to Markdown while preserving structure and formatting.

Usage:
    python pdf_to_markdown.py input.pdf                  # outputs input.md
    python pdf_to_markdown.py input.pdf -o output.md     # custom output path
    python pdf_to_markdown.py input.pdf --ocr            # enable OCR for scanned PDFs
    python pdf_to_markdown.py ./pdfs/                    # convert all PDFs in dir (outputs alongside)
    python pdf_to_markdown.py ./pdfs/ -o ./markdown/     # convert all PDFs, save to output dir

Requirements:
    pip install pdfplumber --break-system-packages
    pip install pytesseract pdf2image --break-system-packages   # only for --ocr flag
"""

import argparse
import re
import sys
from pathlib import Path
from collections import Counter


# ── helpers ──────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalise whitespace and remove stray control characters."""
    if not text:
        return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def table_to_markdown(table: list[list]) -> str:
    """Convert a pdfplumber table (list of rows) to a GFM Markdown table."""
    if not table or not table[0]:
        return ""

    rows = []
    for row in table:
        rows.append([clean_text(str(cell)) if cell is not None else "" for cell in row])

    col_count = max(len(r) for r in rows)
    rows = [r + [""] * (col_count - len(r)) for r in rows]

    widths = [max(len(rows[r][c]) for r in range(len(rows))) for c in range(col_count)]
    widths = [max(w, 3) for w in widths]

    def fmt_row(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    lines = [fmt_row(rows[0])]
    lines.append("| " + " | ".join("-" * w for w in widths) + " |")
    for row in rows[1:]:
        lines.append(fmt_row(row))

    return "\n".join(lines)


def estimate_body_size(pages) -> float:
    """Find the most common font size across all pages — that's the body size."""
    counter: Counter = Counter()
    for page in pages:
        for char in (page.chars or []):
            s = round(float(char.get("size", 0)), 1)
            if s > 0:
                counter[s] += 1
    return counter.most_common(1)[0][0] if counter else 12.0


def determine_heading_level(size: float, body_size: float) -> int | None:
    """Map a font size to a Markdown heading level (1–4) or None for body text."""
    if size <= body_size * 1.05:
        return None
    elif size >= body_size * 2.0:
        return 1
    elif size >= body_size * 1.6:
        return 2
    elif size >= body_size * 1.3:
        return 3
    elif size >= body_size * 1.1:
        return 4
    return None


def group_into_lines(words: list[dict], y_tolerance: float = 4.0) -> list[list[dict]]:
    """Group words into lines by vertical proximity, sorted left→right within each line."""
    if not words:
        return []
    words_by_top = sorted(words, key=lambda w: w["top"])
    lines: list[list[dict]] = []
    current: list[dict] = [words_by_top[0]]

    for word in words_by_top[1:]:
        avg_top = sum(w["top"] for w in current) / len(current)
        if abs(word["top"] - avg_top) <= y_tolerance:
            current.append(word)
        else:
            lines.append(sorted(current, key=lambda w: w["x0"]))
            current = [word]

    lines.append(sorted(current, key=lambda w: w["x0"]))
    return lines


def detect_columns(words: list[dict], page_width: float) -> list[tuple[float, float]]:
    """
    Detect column layout. Returns list of (x_start, x_end) column boundaries.
    For single-column pages returns [(0, page_width)].
    Uses a density histogram to find an empty vertical strip in the middle.
    """
    if not words:
        return [(0, page_width)]

    n_buckets = 30
    bucket_w = page_width / n_buckets
    buckets = [0] * n_buckets
    for w in words:
        b = min(int(w["x0"] / bucket_w), n_buckets - 1)
        buckets[b] += 1

    # Look for a zero-density gap in the middle 25–75% of the page
    mid_lo = int(n_buckets * 0.25)
    mid_hi = int(n_buckets * 0.75)

    for i in range(mid_lo, mid_hi):
        if buckets[i] == 0:
            left_ok  = any(buckets[j] > 0 for j in range(0, i))
            right_ok = any(buckets[j] > 0 for j in range(i + 1, n_buckets))
            if left_ok and right_ok:
                split_x = (i + 0.5) * bucket_w
                return [(0, split_x), (split_x, page_width)]

    return [(0, page_width)]


def assign_column(word: dict, columns: list[tuple[float, float]]) -> int:
    """Return the index of the column this word belongs to."""
    for i, (x_start, _) in enumerate(columns):
        if word["x0"] >= x_start - 5:
            col = i
    return col  # type: ignore[return-value]  # always set because columns starts at 0


def is_noise(line_words: list[dict], page_width: float) -> bool:
    """
    Return True if a line looks like noise:
    - very short text near the page margins (page numbers, watermarks)
    - single-character words stacked near the left edge (arxiv submission watermark)
    """
    text = " ".join(w["text"] for w in line_words).strip()

    # Single/double char near margins
    if len(text) <= 2:
        x0 = line_words[0]["x0"]
        x1 = line_words[-1]["x1"]
        if x0 < 40 or x1 > page_width - 40:
            return True

    # All single-char words near the left margin (arxiv rotated watermark)
    if all(len(w["text"]) == 1 for w in line_words):
        x0 = line_words[0]["x0"]
        x1 = line_words[-1]["x1"]
        if x0 < 50 or x1 > page_width - 50:
            return True

    return False


# ── main conversion ───────────────────────────────────────────────────────────

def pdf_to_markdown(pdf_path: str, ocr: bool = False) -> str:
    """
    Convert *pdf_path* to a Markdown string.
    The H1 title is always derived from the PDF filename stem, not metadata.
    """
    import pdfplumber

    out: list[str] = []

    # H1 title from filename
    out.append(f"# {Path(pdf_path).stem}\n\n")

    with pdfplumber.open(pdf_path) as pdf:
        meta = pdf.metadata or {}
        if meta.get("Author"):
            out.append(f"*Author: {clean_text(meta['Author'])}*\n\n")
        out.append("---\n\n")

        body_size = estimate_body_size(pdf.pages)

        for page in pdf.pages:
            page_w = float(page.width)

            # ── tables ─────────────────────────────────────────────────────
            tables = page.find_tables()
            table_bboxes = [t.bbox for t in tables]
            table_md_map: dict[tuple, str] = {}
            for t in tables:
                data = t.extract()
                if data:
                    table_md_map[t.bbox] = table_to_markdown(data)

            def in_table(x0, top, x1, bottom) -> bool:
                for tb in table_bboxes:
                    if x0 >= tb[0] - 2 and top >= tb[1] - 2 and x1 <= tb[2] + 2 and bottom <= tb[3] + 2:
                        return True
                return False

            # ── words ──────────────────────────────────────────────────────
            # extract_words handles word-boundary spacing correctly
            raw_words = page.extract_words(
                extra_attrs=["fontname", "size"],
                keep_blank_chars=False,
                x_tolerance=3,
                y_tolerance=3,
            )
            words = [
                w for w in raw_words
                if not in_table(w["x0"], w["top"], w["x1"], w["bottom"])
            ]

            if not words and not table_md_map:
                if ocr:
                    ocr_text = ocr_page(page)
                    if ocr_text:
                        out.append(ocr_text + "\n\n")
                continue

            # ── column detection ───────────────────────────────────────────
            columns = detect_columns(words, page_w)

            # Split words into columns, then group each column into lines
            col_buckets: list[list[dict]] = [[] for _ in columns]
            for w in words:
                col_buckets[assign_column(w, columns)].append(w)

            # Build a flat ordered list of line-dicts: one per text line
            # Columns are processed top-to-bottom independently, then appended
            # in order so text reads left-column-first.
            line_records: list[dict] = []
            for col_idx, col_words in enumerate(col_buckets):
                for line_words in group_into_lines(col_words, y_tolerance=4.0):
                    if not line_words or is_noise(line_words, page_w):
                        continue

                    text = clean_text(" ".join(w["text"] for w in line_words))
                    if not text:
                        continue

                    sizes = [float(w["size"]) for w in line_words if w.get("size")]
                    size = max(set(sizes), key=sizes.count) if sizes else body_size

                    line_records.append({
                        "text": text,
                        "size": size,
                        "col": col_idx,
                        "top": line_words[0]["top"],
                        "bottom": max(w["bottom"] for w in line_words),
                    })

            # Within each column sort top→bottom, then append columns in order
            line_records.sort(key=lambda r: (r["col"], r["top"]))

            # ── render: interleave tables and join body lines into paragraphs ─
            pending_tables = sorted(table_md_map.items(), key=lambda kv: kv[0][1])
            tbl_idx = 0

            para: list[str] = []
            prev_bottom = 0.0
            prev_col = 0

            def flush():
                if para:
                    out.append(" ".join(para) + "\n\n")
                    para.clear()

            for rec in line_records:
                # Flush tables that sit above this line
                while tbl_idx < len(pending_tables):
                    tb_bbox, tb_md = pending_tables[tbl_idx]
                    if tb_bbox[1] < rec["top"]:
                        flush()
                        out.append(tb_md + "\n\n")
                        tbl_idx += 1
                    else:
                        break

                heading = determine_heading_level(rec["size"], body_size)

                # Column change or large vertical gap → paragraph break
                if rec["col"] != prev_col:
                    flush()
                elif prev_bottom and (rec["top"] - prev_bottom) > body_size * 1.5:
                    flush()

                prev_bottom = rec["bottom"]
                prev_col = rec["col"]

                if heading:
                    flush()
                    out.append(f"{'#' * heading} {rec['text']}\n\n")
                else:
                    para.append(rec["text"])

            flush()

            # Flush remaining tables at page bottom
            for _, tb_md in pending_tables[tbl_idx:]:
                out.append(tb_md + "\n\n")

    return "".join(out)


# ── OCR fallback ──────────────────────────────────────────────────────────────

def ocr_page(page) -> str:
    """OCR a single pdfplumber page using pytesseract."""
    try:
        import pytesseract
        from PIL import Image
        import io

        img_obj = page.to_image(resolution=200)
        buf = io.BytesIO()
        img_obj.save(buf, format="PNG")
        buf.seek(0)
        img = Image.open(buf)
        return pytesseract.image_to_string(img)
    except ImportError:
        print("⚠️  OCR requested but pytesseract/Pillow not installed.", file=sys.stderr)
        print("    Run: pip install pytesseract pillow --break-system-packages", file=sys.stderr)
        return ""


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert a PDF file (or all PDFs in a directory) to Markdown."
    )
    parser.add_argument("input", help="Path to a PDF file or a directory of PDFs")
    parser.add_argument(
        "-o", "--output",
        help="Output .md file path or output directory (default: same location as input)"
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable OCR for scanned / image-only PDFs (requires pytesseract + pillow)"
    )
    args = parser.parse_args()

    try:
        import pdfplumber  # noqa: F401
    except ImportError:
        print("Error: pdfplumber is not installed.", file=sys.stderr)
        print("Run: pip install pdfplumber --break-system-packages", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if input_path.is_dir():
        pdf_files = sorted(input_path.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in: {input_path}", file=sys.stderr)
            sys.exit(1)

        out_dir = Path(args.output) if args.output else input_path
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.ocr:
            print("OCR mode:   enabled")
        print(f"Found {len(pdf_files)} PDF(s) in: {input_path}")
        print(f"Output dir: {out_dir}\n")

        for pdf_path in pdf_files:
            output_path = out_dir / pdf_path.with_suffix(".md").name
            print(f"Converting: {pdf_path.name} → {output_path.name}")
            try:
                markdown = pdf_to_markdown(str(pdf_path), ocr=args.ocr)
                output_path.write_text(markdown, encoding="utf-8")
            except Exception as exc:
                print(f"  Error: {exc}", file=sys.stderr)

        print("\nDone!")

    else:
        if input_path.suffix.lower() != ".pdf":
            print(f"Warning: File does not have a .pdf extension: {input_path}", file=sys.stderr)

        output_path = Path(args.output) if args.output else input_path.with_suffix(".md")

        print(f"Converting: {input_path}")
        print(f"Output:     {output_path}")
        if args.ocr:
            print("OCR mode:   enabled")

        markdown = pdf_to_markdown(str(input_path), ocr=args.ocr)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"Done! Markdown saved to: {output_path}")


if __name__ == "__main__":
    main()
