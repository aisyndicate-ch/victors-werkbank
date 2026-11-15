"""HTML for the Markdown the werkbank itself writes (reports, evaluations, model
cards): headings, paragraphs, bullet lists, pipe tables and code spans. Every
other construct stays literal text, and all text is escaped, so a hostile
model name or note can only ever appear as text."""

import bisect
import html
import re

HEADING = re.compile(r"(#{1,6}) (.*)")
SEPARATOR = re.compile(r"\|(\s*:?-+:?\s*\|)+")
BACKTICKS = re.compile(r"`+")


def inline(text: str) -> str:
    """Escaped text with code spans: a run of N backticks opens a span that the
    next run of exactly N backticks closes; an unmatched run is literal. Each
    run is looked up once, so hostile text costs n log n, never n squared."""
    runs = [(found.start(), found.end()) for found in BACKTICKS.finditer(text)]
    by_length: dict[int, list[int]] = {}
    for index, (start, end) in enumerate(runs):
        by_length.setdefault(end - start, []).append(index)
    out: list[str] = []
    position = index = 0
    while index < len(runs):
        start, end = runs[index]
        same = by_length[end - start]
        later = bisect.bisect_right(same, index)
        if later == len(same):
            index += 1
            continue
        closing = same[later]
        content = text[end : runs[closing][0]]
        if len(content) > 1 and content[0] == content[-1] == " " and content.strip():
            content = content[1:-1]
        out.append(html.escape(text[position:start]))
        out.append(f"<code>{html.escape(content)}</code>")
        position = runs[closing][1]
        index = closing + 1
    out.append(html.escape(text[position:]))
    return "".join(out)


def cells(line: str) -> list[str]:
    """A table row's cells, split at unescaped pipes; `\\|` is a pipe."""
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|") and not body.endswith("\\|"):
        body = body[:-1]
    parts = re.split(r"(?<!\\)\|", body)
    return [part.strip().replace("\\|", "|") for part in parts]


def _table(header: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{inline(cell)}</th>" for cell in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{inline(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        f'<div class="table"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def to_html(text: str) -> str:
    """HTML for `text`; nothing in it is ever interpreted as markup."""
    lines = text.splitlines()
    out: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            out.append(f"<p>{inline(' '.join(paragraph))}</p>")
            paragraph.clear()

    index = 0
    if lines and lines[0] == "---" and "---" in lines[1:]:
        # Front matter of a dataset card: shown as it is.
        end = lines.index("---", 1)
        out.append(f"<pre>{html.escape(chr(10).join(lines[: end + 1]))}</pre>")
        index = end + 1
    while index < len(lines):
        line = lines[index]
        heading = HEADING.fullmatch(line)
        if not line.strip():
            flush()
            index += 1
        elif heading:
            flush()
            level = len(heading.group(1))
            out.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            index += 1
        elif (
            line.startswith("|")
            and index + 1 < len(lines)
            and SEPARATOR.fullmatch(lines[index + 1].strip())
        ):
            flush()
            header = cells(line)
            index += 2
            rows = []
            while index < len(lines) and lines[index].startswith("|"):
                rows.append(cells(lines[index]))
                index += 1
            out.append(_table(header, rows))
        elif line.startswith("- "):
            flush()
            items: list[str] = []
            while index < len(lines) and (
                lines[index].startswith("- ")
                or (items and lines[index].startswith("  ") and lines[index].strip())
            ):
                if lines[index].startswith("- "):
                    items.append(lines[index][2:])
                else:
                    items[-1] += " " + lines[index].strip()
                index += 1
            out.append(
                "<ul>" + "".join(f"<li>{inline(item)}</li>" for item in items) + "</ul>"
            )
        else:
            paragraph.append(line.strip())
            index += 1
    flush()
    return "\n".join(out)
