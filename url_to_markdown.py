#!/usr/bin/env python3
"""
url_to_markdown.py
------------------
Convert a blog / article URL to a clean Markdown file that preserves
the content hierarchy (headings, lists, code blocks, tables, quotes).

Usage:
    python url_to_markdown.py https://example.com/blog/post
    python url_to_markdown.py https://example.com/blog/post -o output.md
    python url_to_markdown.py https://example.com/blog/post --no-images

Requirements:
    pip install requests beautifulsoup4 --break-system-packages
    pip install lxml --break-system-packages   # optional but faster HTML parser
"""

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse


# ── skip rules ────────────────────────────────────────────────────────────────

SKIP_TAGS = {
    "script", "style", "nav", "footer", "header", "aside",
    "noscript", "iframe", "form", "button", "select", "input",
    "meta", "link", "svg", "canvas",
}

_SKIP_PATTERN = re.compile(
    r"(nav|menu|sidebar|footer|header|banner|cookie|popup|modal|"
    r"share|social|comment|related|recommend|newsletter|subscribe|"
    r"breadcrumb|pagination|tag-list|label|advert|sponsor)",
    re.IGNORECASE,
)


def _should_skip(tag) -> bool:
    if tag.name in SKIP_TAGS:
        return True
    classes = " ".join(tag.get("class", []))
    tag_id = tag.get("id", "")
    return bool(_SKIP_PATTERN.search(classes) or _SKIP_PATTERN.search(tag_id))


# ── content area detection ────────────────────────────────────────────────────

_CONTENT_SELECTORS = [
    "article",
    "main",
    "[role=main]",
    ".post-content",
    ".article-content",
    ".entry-content",
    ".post-body",
    ".blog-post",
    ".content-body",
    "#content",
    "#main",
    ".content",
]


def _find_main(soup):
    for sel in _CONTENT_SELECTORS:
        el = soup.select_one(sel)
        if el:
            return el
    return soup.find("body") or soup


# ── table helper ──────────────────────────────────────────────────────────────

def _table_to_md(table_node) -> str:
    rows = []
    for tr in table_node.find_all("tr"):
        cells = [_clean(cell.get_text()) for cell in tr.find_all(["th", "td"])]
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    col_count = max(len(r) for r in rows)
    rows = [r + [""] * (col_count - len(r)) for r in rows]
    widths = [max(len(rows[i][c]) for i in range(len(rows))) for c in range(col_count)]
    widths = [max(w, 3) for w in widths]

    def fmt(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    lines = [fmt(rows[0]), "| " + " | ".join("-" * w for w in widths) + " |"]
    lines += [fmt(r) for r in rows[1:]]
    return "\n".join(lines)


# ── text helpers ──────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ── DOM → Markdown ────────────────────────────────────────────────────────────

def _node_to_md(node, base_url: str = "", list_depth: int = 0) -> str:
    from bs4 import NavigableString, Tag

    if isinstance(node, NavigableString):
        return re.sub(r"[\r\n\t ]+", " ", str(node))

    if not isinstance(node, Tag):
        return ""

    name = node.name.lower() if node.name else ""

    if _should_skip(node):
        return ""

    def children_md(depth=list_depth):
        return "".join(_node_to_md(c, base_url, depth) for c in node.children)

    # ── headings ──────────────────────────────────────────────────────────────
    if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        inner = _clean(node.get_text())
        return f"\n\n{'#' * int(name[1])} {inner}\n\n" if inner else ""

    # ── paragraph ─────────────────────────────────────────────────────────────
    if name == "p":
        inner = _clean(children_md())
        return f"\n\n{inner}\n\n" if inner else ""

    # ── blockquote ────────────────────────────────────────────────────────────
    if name == "blockquote":
        inner = children_md().strip()
        quoted = "\n".join(f"> {line}" for line in inner.splitlines())
        return f"\n\n{quoted}\n\n"

    # ── fenced code block ─────────────────────────────────────────────────────
    if name == "pre":
        code_tag = node.find("code")
        code_text = (code_tag or node).get_text()
        lang = ""
        if code_tag:
            for cls in code_tag.get("class", []):
                m = re.match(r"(?:language|lang)-(\w+)", cls)
                if m:
                    lang = m.group(1)
                    break
        return f"\n\n```{lang}\n{code_text.rstrip()}\n```\n\n"

    # ── inline code ───────────────────────────────────────────────────────────
    if name == "code":
        return f"`{node.get_text()}`"

    # ── horizontal rule ───────────────────────────────────────────────────────
    if name == "hr":
        return "\n\n---\n\n"

    # ── line break ────────────────────────────────────────────────────────────
    if name == "br":
        return "  \n"

    # ── lists ─────────────────────────────────────────────────────────────────
    if name in ("ul", "ol"):
        is_ordered = name == "ol"
        indent = "  " * list_depth
        items = []
        counter = 0
        for child in node.children:
            if hasattr(child, "name") and child.name == "li":
                counter += 1
                inner = "".join(
                    _node_to_md(c, base_url, list_depth + 1) for c in child.children
                ).strip()
                inner = re.sub(r"\n{2,}", "\n", inner)
                bullet = f"{counter}." if is_ordered else "-"
                items.append(f"{indent}{bullet} {inner}")
        return "\n\n" + "\n".join(items) + "\n\n" if items else ""

    # ── table ─────────────────────────────────────────────────────────────────
    if name == "table":
        md = _table_to_md(node)
        return f"\n\n{md}\n\n" if md else ""

    # ── image ─────────────────────────────────────────────────────────────────
    if name == "img":
        alt = node.get("alt", "")
        src = node.get("src", "")
        if src and base_url:
            src = urljoin(base_url, src)
        return f"![{alt}]({src})" if src else ""

    # ── link ──────────────────────────────────────────────────────────────────
    if name == "a":
        inner = children_md().strip()
        href = node.get("href", "")
        if href and base_url:
            href = urljoin(base_url, href)
        if not inner:
            return ""
        return f"[{inner}]({href})" if href else inner

    # ── bold ──────────────────────────────────────────────────────────────────
    if name in ("strong", "b"):
        inner = children_md().strip()
        return f"**{inner}**" if inner else ""

    # ── italic ────────────────────────────────────────────────────────────────
    if name in ("em", "i"):
        inner = children_md().strip()
        return f"*{inner}*" if inner else ""

    # ── strikethrough ─────────────────────────────────────────────────────────
    if name in ("s", "del", "strike"):
        inner = children_md().strip()
        return f"~~{inner}~~" if inner else ""

    # ── div / section / span / everything else ────────────────────────────────
    return children_md()


# ── main conversion ───────────────────────────────────────────────────────────

def url_to_markdown(url: str, include_images: bool = True) -> str:
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding

    try:
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception:
        soup = BeautifulSoup(resp.text, "html.parser")

    # ── metadata ──────────────────────────────────────────────────────────────
    title = ""
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = _clean(og_title["content"])
    elif soup.find("title"):
        title = _clean(soup.find("title").get_text())

    description = ""
    og_desc = (
        soup.find("meta", property="og:description")
        or soup.find("meta", attrs={"name": "description"})
    )
    if og_desc and og_desc.get("content"):
        description = _clean(og_desc["content"])

    author = ""
    for sel in [
        '[name="author"]', '[property="article:author"]',
        ".author", ".byline", ".post-author",
    ]:
        el = soup.select_one(sel)
        if el:
            author = _clean(el.get("content") or el.get_text())
            if author:
                break

    published = ""
    for sel in [
        '[property="article:published_time"]',
        'time[datetime]',
        ".date", ".published", ".post-date",
    ]:
        el = soup.select_one(sel)
        if el:
            published = el.get("content") or el.get("datetime") or _clean(el.get_text())
            if published:
                break

    # ── find & render main content ────────────────────────────────────────────
    main = _find_main(soup)
    parsed = urlparse(url)
    base_url = f"{parsed.scheme}://{parsed.netloc}" if include_images else ""

    content_md = _node_to_md(main, base_url=base_url)
    content_md = re.sub(r"\n{3,}", "\n\n", content_md).strip()

    # ── assemble document ─────────────────────────────────────────────────────
    parts: list[str] = []

    if title:
        parts.append(f"# {title}\n")

    meta_lines = []
    if author:
        meta_lines.append(f"*Author: {author}*")
    if published:
        meta_lines.append(f"*Published: {published}*")
    meta_lines.append(f"*Source: <{url}>*")
    parts.append("  \n".join(meta_lines))
    parts.append("\n---\n")

    if description:
        parts.append(f"> {description}\n")

    parts.append(content_md)

    return "\n".join(parts)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert a blog/article URL to a clean Markdown file."
    )
    parser.add_argument("url", help="URL of the blog post or article")
    parser.add_argument(
        "-o", "--output",
        help="Output .md file path (default: derived from the URL slug)"
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Omit image references from the output"
    )
    args = parser.parse_args()

    # dependency check
    missing = []
    try:
        import requests  # noqa: F401
    except ImportError:
        missing.append("requests")
    try:
        from bs4 import BeautifulSoup  # noqa: F401
    except ImportError:
        missing.append("beautifulsoup4")

    if missing:
        pkgs = " ".join(missing)
        print(f"Error: missing packages: {pkgs}", file=sys.stderr)
        print(f"Run: pip install {pkgs} --break-system-packages", file=sys.stderr)
        sys.exit(1)

    # output path
    if args.output:
        output_path = Path(args.output)
    else:
        parsed = urlparse(args.url)
        slug = parsed.path.rstrip("/").split("/")[-1] or parsed.netloc.replace(".", "_")
        slug = re.sub(r"[^\w-]", "_", slug)
        output_path = Path(f"{slug}.md")

    print(f"Fetching:  {args.url}")
    print(f"Output:    {output_path}")

    try:
        markdown = url_to_markdown(args.url, include_images=not args.no_images)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    output_path.write_text(markdown, encoding="utf-8")
    print(f"Done! Markdown saved to: {output_path}")


if __name__ == "__main__":
    main()
