"""The privacy check over the exact member bytes of a submission, the members
of each `run.tar` included.

It first removes the fixed links the build writes into its own pages and the
result page's stylesheet, and reads the result page as its text: without its
tags, whose words are the page's own markup (every value on the page is
escaped text), and with its character references resolved. In the
always-sent part (`submission.json`, `evaluation/`, `result/`) it looks for
URLs, absolute local paths, drive letters, network shares, `file:`, the word
`localhost`, IPv4 addresses in a loopback, private, link-local or shared range
and a host name with a port; a path, drive letter, share, address or host with
port counts only at the start of a token. Everywhere it looks for this
computer's local values as whole words, ignoring case: host name, user name,
home and working directory, the submitted directories, the endpoint and, where
`WERKBANK_API_KEY` is set when the submission is built, the API key. Short
values, generic names and words of the build's own fixed vocabulary are
skipped, and so are task files identical to the build's own task packages; an
API key is sought even where it is a generic name or such a word. In
the raw data a container image reference that names a registry host or a
namespace is a finding, unless it is exactly the build's default image. A
finding names the member, the line and the kind, never the matched value.
"""

import ast
import getpass
import html
import io
import ipaddress
import json
import os
import platform
import pwd
import re
import socket
import tarfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Iterable, Sequence
from urllib.parse import urlsplit

from victors_werkbank.agent.episode import harness_files
from victors_werkbank.agent.transport import (
    API_KEY_VARIABLE,
    SECRET_KEY_CHARS,
    holds_key,
    strict_json,
)
from victors_werkbank.harbor_smoke import SMOKE_IMAGE
from victors_werkbank.integration.tasks import PACKAGE, TASKS
from victors_werkbank.registry import CASE_LIMITATIONS, CASES
from victors_werkbank.results import comparison, evaluation, export, outcomes, tables
from victors_werkbank.results.evaluation_check import CATEGORIES
from victors_werkbank.serving import RUNTIMES
from werkbank_result import page, writer

from . import archive, texts

# Links the build writes into its own pages, removed before the check.
FIXED_LINKS: tuple[bytes, ...] = ()
# The result page's stylesheet, removed before the check like the links (its
# line breaks kept, so a finding's line stays the page's line): its words are
# no text, so they never stand in the vocabulary.
STYLESHEET = page.STYLE.encode("utf-8")
# The result page's member, read as its text.
PAGE_MEMBER = f"result/{writer.PAGE}"
_TAG = re.compile(rb"<[^<>\n]*>")
GENERIC = frozenset(
    {
        "localhost",
        "ubuntu",
        "debian",
        "fedora",
        "raspberrypi",
        "root",
        "user",
        "admin",
        "runner",
    }
)
MIN_LENGTH = 3
# The kind of the API key among the local values: sought whatever its value.
KEY_KIND = "an API key"
# The modules whose string constants, docstrings aside, are the texts written
# into an evaluation's files; the result page's texts are `page_texts()`.
WRITTEN_MODULES: tuple[ModuleType, ...] = (
    evaluation,
    tables,
    comparison,
    outcomes,
)
# The fields of a result page's `result.json`; its repeat records' fields and
# kinds are `writer.REPEAT_FIELDS` and `writer.REPEAT_KINDS`.
RESULT_FIELDS = (
    "format",
    "evaluation",
    "sha256",
    "renderer_sha256",
    "repeats",
    "files",
)

# Every member outside `raw/` is always sent.
RAW = "raw/"
RUN_ARCHIVE = "run.tar"
PLAN = "plan.json"
TASK_DIRECTORY = "tasks/"
HARNESS_DIRECTORY = "tests/domain/victors_werkbank/"

_START = rb"(?<![A-Za-z0-9./-])"
_WORD_BEFORE = rb"(?<![A-Za-z0-9_])"
_WORD_AFTER = rb"(?![A-Za-z0-9_])"
PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("a URL", re.compile(rb"[A-Za-z][A-Za-z0-9+.-]*://")),
    (
        "a local path",
        re.compile(
            _START
            + rb"(?:/(?:home|Users|root|mnt|media|srv|opt|var|tmp|run|Volumes)/|~/)"
        ),
    ),
    ("a drive letter", re.compile(_START + rb"[A-Za-z]:[\\/]")),
    ("a network share", re.compile(_START + rb"\\\\[^\\\s]+\\")),
    ("a file reference", re.compile(_START + rb"file:")),
    ("localhost", re.compile(_WORD_BEFORE + rb"localhost" + _WORD_AFTER, re.I)),
)
_IPV4 = re.compile(
    _START + rb"([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})(?![0-9])"
)
PRIVATE_NETWORKS = tuple(
    ipaddress.IPv4Network(network)
    for network in (
        "127.0.0.0/8",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "169.254.0.0/16",
        "100.64.0.0/10",
    )
)
_HOST_PORT = re.compile(_START + rb"[A-Za-z][A-Za-z0-9.-]*:([0-9]{1,5})" + _WORD_AFTER)
IMAGE_FINDING = "an image reference that names a registry host or namespace"
_WORDS = re.compile(r"[a-z0-9_]+")
_COMPOUNDS = re.compile(r"[a-z0-9_]+(?:[.-][a-z0-9_]+)+")


@dataclass(frozen=True)
class Finding:
    """Where something identifying was found and what kind it is."""

    member: str
    line: int
    kind: str
    inner: str | None = None

    def text(self) -> str:
        where = self.member if self.inner is None else f"{self.member}: {self.inner}"
        return f"{where}, line {self.line}: {self.kind}"


def local_values(
    directories: Iterable[Path], endpoint: str | None = None
) -> list[tuple[str, str]]:
    """(kind, value) of this computer's local values: host name full and short
    (never a DNS lookup), user name, home and working directory, the
    submitted directories, where known the endpoint address and host, and
    the API key where `WERKBANK_API_KEY` is set in this process."""
    found: list[tuple[str, str]] = []
    for name in (socket.gethostname(), platform.node()):
        found += [("a host name", name), ("a host name", name.split(".")[0])]
    try:
        found.append(("a user name", getpass.getuser()))
    except (KeyError, OSError):
        pass
    try:
        found.append(("a user name", pwd.getpwuid(os.getuid()).pw_name))
    except KeyError:
        pass
    try:
        found.append(("the home directory", str(Path.home())))
    except (KeyError, OSError, RuntimeError):
        pass
    try:
        found.append(("the working directory", os.getcwd()))
    except OSError:
        pass
    for directory in directories:
        found.append(("a submitted directory", str(directory.absolute())))
        try:
            found.append(("a submitted directory", str(directory.resolve())))
        except (OSError, RuntimeError):
            pass
    if endpoint:
        found.append(("the endpoint address", endpoint))
        parts = urlsplit(endpoint)
        if parts.netloc:
            found.append(("the endpoint address", parts.netloc))
        try:
            host = parts.hostname
        except ValueError:
            host = None
        if host:
            found.append(("the endpoint host", host))
    # A shorter key is a placeholder, not a secret, and is not sought.
    key = os.environ.get(API_KEY_VARIABLE)
    if key and len(key) >= SECRET_KEY_CHARS:
        found.append((KEY_KIND, key))
    return found


def _constants(module: ModuleType) -> list[str]:
    """Every string constant in the source of `module` but its docstrings."""
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    documented = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, documented)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for pair in value.items() for text in _strings(list(pair))]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _strings(item)]
    return []


def page_texts() -> list[str]:
    """The result page's fixed texts: the values of its module-level
    constants, the texts it shares with other modules (`page.SHARED_TEXTS`)
    among them. Never its stylesheet, its markup or the record keys it reads,
    which are no text on the page."""
    return [
        text
        for name, value in vars(page).items()
        if name.isupper() and name != "STYLE"
        for text in _strings(value)
    ]


def _keys(value: object) -> list[str]:
    if isinstance(value, dict):
        return [key for key, item in value.items() for key in [key, *_keys(item)]]
    if isinstance(value, list):
        return [key for item in value for key in _keys(item)]
    return []


def identifiers() -> list[str]:
    """The submission's own identifiers: the envelope's field names, format
    IDs, question fields and choice IDs, the archive's member names and the
    result record's format, file and field names with its repeat records'
    fields and kinds."""
    sample = archive.envelope(
        installation_id="0" * 32,
        evaluation={
            "format": "",
            "inputs": [{"run_id": "", "plan_sha256": "", "run_sha256": ""}],
        },
        evaluation_sha256="",
        case_set={"name": None, "case_ids": []},
        answers={},
        raw_data=False,
        members=[("result/x", b"")],
    )
    found = _keys(json.loads(sample))
    found += [texts.SUBMISSION_FORMAT, texts.QUESTIONS_FORMAT, texts.CONSENT_FORMAT]
    found += [choice for _, _, choices in texts.QUESTIONS for choice, _ in choices]
    found += [archive.ENVELOPE, *archive.TOP, export.INDEX, export.ARCHIVE]
    found += [writer.RESULT_FORMAT, writer.PAGE, writer.RESULT, *RESULT_FIELDS]
    found += [*writer.REPEAT_FIELDS, *writer.REPEAT_KINDS]
    return found


@cache
def vocabulary() -> frozenset[str]:
    """The words the build writes into a submission itself, in lower case
    (single words and words joined by `.` or `-`): the texts of the
    evaluation's files and the result page (headings, table headers, notices,
    limitations), the category names, the case IDs, the engine names and the
    submission's own identifiers. Never the client's dialog: its consent
    text, prompts and messages are not sent; never the result page's
    stylesheet or markup, which the check removes."""
    found: list[str] = []
    for module in WRITTEN_MODULES:
        found += _constants(module)
    found += page_texts() + tables.NOTICE + evaluation.LIMITATIONS
    found += [text for items in CASE_LIMITATIONS.values() for text in items]
    found += list(CATEGORIES) + list(CASES)
    found += [name for name, _ in RUNTIMES.values()] + list(RUNTIMES)
    found += identifiers()
    words: set[str] = set()
    for text in found:
        lower = text.lower()
        words.update(_WORDS.findall(lower))
        words.update(_COMPOUNDS.findall(lower))
    return frozenset(words)


def usable(
    values: Iterable[tuple[str, str]], words: frozenset[str]
) -> list[tuple[str, str]]:
    """The values the check looks for: none shorter than three characters,
    no generic name and no word of the fixed vocabulary; each once. An API
    key is kept even where it is such a name or word: a submission that
    holds it is never sent, whatever the key is."""
    kept: dict[str, tuple[str, str]] = {}
    for kind, value in values:
        lower = value.lower()
        if len(value) < MIN_LENGTH:
            continue
        if kind != KEY_KIND and (lower in GENERIC or lower in words):
            continue
        kept.setdefault(lower, (kind, value))
    return list(kept.values())


class _Values:
    """One pattern for the local values, longest first, and each matched
    spelling's kind; a value is also sought as JSON writes it. An API key is
    sought apart, in every line, wherever it stands and however its
    characters are escaped (`holds_key`): it is a secret, not a name, so it
    needs no word around it."""

    def __init__(self, values: Sequence[tuple[str, str]]):
        self.kinds: dict[bytes, str] = {}
        self.keys = [value for kind, value in values if kind == KEY_KIND]
        for kind, value in values:
            if kind == KEY_KIND:
                continue
            forms = {value.encode("utf-8"), json.dumps(value)[1:-1].encode("ascii")}
            for form in forms:
                self.kinds.setdefault(form.lower(), kind)
        ordered = sorted(self.kinds, key=lambda form: (-len(form), form))
        self.pattern = (
            re.compile(
                _WORD_BEFORE
                + rb"(?:"
                + b"|".join(re.escape(form) for form in ordered)
                + rb")"
                + _WORD_AFTER,
                re.I,
            )
            if ordered
            else None
        )

    def findings(self, data: bytes) -> list[tuple[int, str]]:
        found = []
        if self.pattern is not None:
            found = [
                (match.start(), self.kinds[match.group().lower()])
                for match in self.pattern.finditer(data)
            ]
        if self.keys:
            position = 0
            for line in data.splitlines(keepends=True):
                if any(holds_key(line, key) for key in self.keys):
                    found.append((position, KEY_KIND))
                position += len(line)
        return found


def _line(data: bytes, position: int) -> int:
    return data.count(b"\n", 0, position) + 1


def _patterns(data: bytes) -> list[tuple[int, str]]:
    found = [
        (match.start(), kind)
        for kind, pattern in PATTERNS
        for match in pattern.finditer(data)
    ]
    for match in _IPV4.finditer(data):
        octets = [int(part) for part in match.groups()]
        if all(octet <= 255 for octet in octets):
            address = ipaddress.IPv4Address(".".join(map(str, octets)))
            if any(address in network for network in PRIVATE_NETWORKS):
                found.append((match.start(), "a private IPv4 address"))
    for match in _HOST_PORT.finditer(data):
        if 1 <= int(match.group(1)) <= 65535:
            found.append((match.start(), "a host name with a port"))
    return found


def _without_links(data: bytes) -> bytes:
    """`data` without the fixed links and the result page's stylesheet, whose
    line breaks stay."""
    for link in FIXED_LINKS:
        data = data.replace(link, b"")
    return data.replace(STYLESHEET, b"\n" * STYLESHEET.count(b"\n"))


def _page_text(data: bytes) -> bytes:
    """The result page as its text: each tag a space, so the words of two
    elements stay apart, and its character references resolved, line for
    line. A page that is no UTF-8 is searched as it stands."""
    try:
        shown = _TAG.sub(b" ", data).decode("utf-8")
    except UnicodeDecodeError:
        return data
    return html.unescape(shown).encode("utf-8")


@cache
def _harness() -> frozenset[Path]:
    return frozenset(harness_files())


@cache
def _package_file(member: str) -> bytes | None:
    """The bytes this build ships for a run's task file `tasks/<name>/<rel>`,
    or None: its committed task package file or the harness file a task
    carries under `tests/domain/victors_werkbank/`."""
    name, _, rel = member.removeprefix(TASK_DIRECTORY).partition("/")
    if not archive.plain_path(name) or not archive.plain_path(rel):
        return None
    if rel.startswith(HARNESS_DIRECTORY):
        path = PACKAGE / rel.removeprefix(HARNESS_DIRECTORY)
        if path not in _harness():
            return None
    else:
        path = TASKS / name / rel
    try:
        return path.read_bytes() if path.is_file() and not path.is_symlink() else None
    except OSError:
        return None


def _image_names_host(plan: bytes) -> bool:
    """Whether a run's recorded image reference names a registry host or a
    namespace: a `/` before `@sha256:`; an unreadable plan counts as one. The
    build's default image, whole and exact, does not count: every tester's run
    records that same reference, so it says nothing about the tester."""
    try:
        image = strict_json(plan.decode("utf-8"))["image"]
    except (ValueError, UnicodeError, TypeError, KeyError):
        return True
    if image == SMOKE_IMAGE:
        return False
    return not isinstance(image, str) or "/" in image.partition("@sha256:")[0]


def _run_archive(member: str, data: bytes, values: _Values) -> list[Finding]:
    found: list[Finding] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as run:
            for info in run:
                if info.isdir():
                    continue
                stream = run.extractfile(info) if info.type == tarfile.REGTYPE else None
                if stream is None:
                    found.append(Finding(member, 1, "a member that is no file"))
                    continue
                content = _without_links(stream.read())
                if info.name == PLAN and _image_names_host(content):
                    found.append(Finding(member, 1, IMAGE_FINDING, info.name))
                if (
                    info.name.startswith(TASK_DIRECTORY)
                    and _package_file(info.name) == content
                ):
                    continue
                found += [
                    Finding(member, _line(content, position), kind, info.name)
                    for position, kind in values.findings(content)
                ]
    except tarfile.TarError:
        found.append(Finding(member, 1, "an unreadable run archive"))
    return found


def check(
    members: Sequence[tuple[str, bytes]], values: Sequence[tuple[str, str]]
) -> tuple[list[Finding], list[Finding]]:
    """(findings in the always-sent part, findings in the raw data) of a
    submission's `members`, for the usable local `values`."""
    local = _Values(values)
    always: list[Finding] = []
    raw: list[Finding] = []
    for member, original in members:
        data = _without_links(original)
        if member == PAGE_MEMBER:
            data = _page_text(data)
        if not member.startswith(RAW):
            found = sorted(_patterns(data) + local.findings(data))
            always += [Finding(member, _line(data, p), kind) for p, kind in found]
        elif member.endswith("/" + RUN_ARCHIVE):
            raw += _run_archive(member, data, local)
        else:
            raw += [
                Finding(member, _line(data, position), kind)
                for position, kind in local.findings(data)
            ]
    return always, raw
