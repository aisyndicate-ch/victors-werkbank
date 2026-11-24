"""The steps of `werkbank start`, in order: readiness, the server, the model,
the server check, the mode, its number of passes and the cap, the passes
(three, or one: the trial's, or on request a benchmark mode's), the evaluation,
the result page and, after three passes, the offer to submit.

Exit codes: 0 every pass measured completely and the page written; 3 the
series is evaluated but a pass left cases unmeasured; 2 refused or failed; 1
stopped by the tester. Submitting or declining never changes the code.

An API key (`WERKBANK_API_KEY`) goes only to an endpoint the tester named,
with `--endpoint` or as a typed base URL: its model list read, the server
check and every pass send it. The search never sends it, and neither does
anything sent afterwards to a server the search found. The key is in no
command line, printed command or file.
"""

import json
import os
import re
import shlex
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from victors_werkbank.agent.measurements import TEMPLATE_KWARGS
from victors_werkbank.agent.model_process import IsolatedEndpoint
from victors_werkbank.agent.transport import API_KEY_VARIABLE, EndpointError
from victors_werkbank.registry import (
    CASES,
    MIN_TOKENS_PER_CALL,
    case_sets,
    context_need,
    context_recommendation,
    module_of,
    scored_sets,
)
from victors_werkbank.serving import NOT_LOOPBACK, PROBE_SECONDS
from victors_werkbank.thinking import RecipeError, resolve

from . import child, discovery
from .console import Console, Stopped, duration, shown

COMPLETE = 0
STOPPED = 1
FAILED = 2
UNMEASURED = 3

SEEDS = (1, 2, 3)
# The trial: one pass over eight easy anchors, one per category, in registry
# order. It checks the installation, the agent loop, the containers and the
# result page; it has no score, compares with nothing and is never submitted.
# The list lives here, outside the harness digest, and is no case set.
TRIAL_SEEDS = (1,)
TRIAL_CASES = ("M04", "S06", "DA05", "R03", "A15", "L01", "C11", "MT01")
TRIAL_TEXT = (
    "A quick check that the installation, the agent loop, the containers and"
    " the result page work, a few minutes; no benchmark: no score, not"
    " comparable and never submitted"
)
# The benchmark modes, each one case set. Quick is a first picture, Normal the
# standard run and Full the one with the hardest tasks; a mode whose set is a
# scored one (`registry.scored_sets`) has an overall score over its own tasks,
# any other benchmark mode none, so the line follows the shipped lists; a
# module's mode says in its own words that it stands outside the score.
QUICK_TEXT = "A quick check on tasks that a 64K context can run, a first picture"
NORMAL_TEXT = "Every task but the hardest ones"
FULL_TEXT = "Every task, the hardest included"
SCORED_TEXT = "an overall score over these tasks"
UNSCORED_TEXT = "no overall score"
# The German module: its own case set outside the score, offered after the
# benchmark modes. Three passes or one, like them,
# and three passes are offered for submission; its page shows the tasks
# solved per pass and the subgoal fulfilment, never points.
GERMAN_SET = "deutsch-v1"
GERMAN_TEXT = (
    "Tasks on German documents outside the score: the tasks solved per pass"
    " and the subgoal fulfilment, no points"
)
# Every mode but the trial runs three passes, or on request one: a first look in
# about a third of the time that is never submitted, as results to send hold
# the three passes.
SINGLE_SEEDS = (1,)
PASSES_QUESTION = "Three passes or one?"
PASSES_TEXTS = (
    "Three passes with seeds 1, 2 and 3: a result to compare, with the range"
    " of the passes, that can be submitted",
    "One pass with seed 1: about a third of the time; it has no score, which"
    " needs three passes, and it is never submitted",
)
# Every pass is a `werkbank run` without an instance value of its own, so
# each draws one, and so does a repeated pass: a given value would keep the
# pass out of the pool.
DRAWN_DATA = (
    "Each pass gets newly drawn data for every task; a repeated pass draws new"
    " data too."
)
SINGLE_SUMMARY = (
    "This was one pass: it is never submitted. For results to send, start"
    " Quick, Normal, Full or German with three passes."
)
# The largest per-response cap a run accepts. The passes use it where the
# server's context covers the chosen mode with it, so a long response is not cut
# short where the context allows it.
LARGEST_CAP = 65536
EVALUATION = "evaluation"
RESULT = "result"
MAX_NAMES = 100
NOT_A_URL = "not a valid URL"
VERDICT = "Ready: "
# What a tester reads about an API key; never its value.
KEY_SENT = f"API key: sent with every request to this server ({API_KEY_VARIABLE})"
KEY_NOT_SENT = (
    f"API key: not sent ({API_KEY_VARIABLE} is set, but a key goes only to a"
    " server you name with --endpoint, never to one the search found)"
)
KEY_NEEDED = (
    "{url} answered but needs a key: name it with --endpoint {url} and set the"
    f" environment variable {API_KEY_VARIABLE} to its key."
)

INSTALL = "./install.sh in the root of the Werkbank checkout (docs/INSTALLATION.md)"
# The fix for each readiness error `harbor_smoke.preflight` names; `{image}`
# is the image reference. The installation script installs the hash-locked
# environment, names what keeps Docker from working for the user and pulls the
# pinned image, so every fix it covers points there.
FIXES: dict[str, tuple[str, ...]] = {
    "Harbor requires Python >=3.12": (
        INSTALL + "; it creates .venv with Python 3.12 or newer. Then start"
        " .venv/bin/werkbank start.",
    ),
    "Install the exact archive in requirements-harbor.txt": (INSTALL,),
    "Install the hash-locked requirements-harbor.txt": (INSTALL,),
    "Docker CLI is unavailable": (
        "Install Docker so that the docker command works for your user"
        " (docs/users/PLATFORMS.md), then " + INSTALL + ".",
    ),
    "Docker access or the explicitly prepared image is missing": (
        INSTALL + "; it checks your access to Docker and pulls the pinned image.",
    ),
    "Docker image inspection failed": (
        "Start Docker and check that it answers: docker info",
    ),
}
# The installation script pulls only the pinned image; an image given with
# --image is pulled by hand.
OTHER_IMAGE_FIXES: dict[str, tuple[str, ...]] = {
    "Docker access or the explicitly prepared image is missing": (
        "docker pull {image}",
        "If Docker itself does not answer, start it and check: docker info",
    ),
}
WSL_NOTE = (
    "Under WSL2, a server running under Windows is not reachable on 127.0.0.1"
    " without mirrored networking: give its address with --endpoint (for"
    " example the Windows host's address)."
)
# The kernel release WSL2 names itself in ("...-microsoft-standard-WSL2").
OSRELEASE = Path("/proc/sys/kernel/osrelease")
LIMA_NOTE = (
    "In a Lima VM (Lima or Colima), a server on the computer that runs it is"
    " reached as host.lima.internal, not 127.0.0.1: for example"
    " http://host.lima.internal:11434/v1 for Ollama on its usual port."
)
ORBSTACK_NOTE = (
    "In an OrbStack machine, a server on the Mac is reached as host.orb.internal,"
    " not 127.0.0.1: for example http://host.orb.internal:11434/v1 for Ollama on"
    " its usual port."
)
# Lima, and Colima built on it, write "<address> host.lima.internal" into the
# guest's hosts file at every boot; an OrbStack machine shows the Mac's files
# at /mnt/mac. Both are only read, and never more than HOSTS_BYTES of the file.
HOSTS = Path("/etc/hosts")
HOSTS_BYTES = 65536
LIMA_LINE = re.compile(r"[0-9A-Fa-f.:]+[ \t]+host\.lima\.internal(?:[ \t]|$)")
MAC_FILES = Path("/mnt/mac")
PATH_HINT = (
    "{count} listed model{s} named by a file or its path cannot be chosen:"
    " serve the model under a name (llama.cpp: --alias NAME; vLLM:"
    " --served-model-name NAME)."
)


@dataclass(frozen=True)
class Mode:
    """What a mode runs: its case list, the case set it is (if any), the
    selection `werkbank run` takes for it, what it is in a line, whether it is
    the trial (one pass, no score and no offer to submit) and whether a
    benchmark mode runs as a single pass (no offer to submit)."""

    name: str
    title: str
    case_ids: tuple[str, ...]
    case_set: str | None
    selection: tuple[str, ...]
    about: str = ""
    trial: bool = False
    single: bool = False

    @property
    def seeds(self) -> tuple[int, ...]:
        """The seeds of its passes, one pass per seed."""
        if self.trial:
            return TRIAL_SEEDS
        return SINGLE_SEEDS if self.single else SEEDS

    @property
    def submitted(self) -> bool:
        """Whether its results are offered for submission: those of three
        passes, never a trial's or a single pass's."""
        return not (self.trial or self.single)

    def passes_text(self) -> str:
        if len(self.seeds) == 1:
            return f"one pass with seed {self.seeds[0]}"
        return "three passes with seeds 1, 2 and 3"

    def count_text(self) -> str:
        return "one pass" if len(self.seeds) == 1 else "three passes"

    def episodes(self) -> int:
        return sum(len(CASES[case_id].branches) for case_id in self.case_ids)

    def scope(self) -> str:
        count = len(self.case_ids)
        what = f"{count} case{'s' if count != 1 else ''}"
        if self.case_set is not None:
            what = f"case set {self.case_set}, {what}"
        episodes = self.episodes()
        return f"{what} ({episodes} episode{'s' if episodes != 1 else ''})"


def _benchmark(name: str, title: str, case_set: str, about: str) -> Mode:
    case_ids = case_sets()[case_set]
    if module_of(case_ids) is None:
        score = SCORED_TEXT if case_set in scored_sets() else UNSCORED_TEXT
        about = f"{about}; {score}"
    return Mode(name, title, case_ids, case_set, ("--set", case_set), about)


def modes() -> dict[str, Mode]:
    return {
        "quick": _benchmark("quick", "Quick", "quick-v5", QUICK_TEXT),
        "normal": _benchmark("normal", "Normal", "normal-v3", NORMAL_TEXT),
        "full": _benchmark("full", "Full", "full-v2", FULL_TEXT),
        "german": _benchmark("german", "German", GERMAN_SET, GERMAN_TEXT),
        "trial": Mode(
            "trial",
            "Trial",
            TRIAL_CASES,
            None,
            tuple(item for case_id in TRIAL_CASES for item in ("--case", case_id)),
            TRIAL_TEXT,
            trial=True,
        ),
    }


@dataclass(frozen=True)
class Options:
    """The command line, every value already checked as `run` checks it."""

    endpoint: str | None = None
    model: str | None = None
    mode: str | None = None
    yes: bool = False
    # The number of passes of a benchmark mode, 1 or 3; None asks, or with
    # --yes runs three.
    passes: int | None = None
    output: Path | None = None
    max_tokens_per_call: int | None = None
    temperature: float | int | None = None
    top_p: float | int | None = None
    top_k: int | None = None
    stream: bool = False
    serving_note: Path | None = None
    note: dict[str, Any] | None = None
    hardware_label: str | None = None
    image: str | None = None
    # The template variables of --chat-template-kwargs as canonical JSON, so
    # the options stay hashable and every pass gets the same text.
    chat_template_kwargs: str | None = None
    # The --thinking level, resolved for the chosen model (`thinking_for`).
    thinking: str | None = None
    # The API key of WERKBANK_API_KEY, sent only to an endpoint the tester
    # named. Never shown: it is in no text of these options, no command line
    # (`passed_on`) and no file.
    api_key: str | None = field(default=None, repr=False)

    def sampling(self, model: str | None = None) -> dict[str, Any]:
        """The settings every request carries; with `model`, a --thinking
        level resolved into its recipe's template variables."""
        found: dict[str, Any] = {
            key: value
            for key, value in (
                ("temperature", self.temperature),
                ("top_p", self.top_p),
                ("top_k", self.top_k),
            )
            if value is not None
        }
        if self.chat_template_kwargs is not None:
            found[TEMPLATE_KWARGS] = json.loads(self.chat_template_kwargs)
        if self.thinking is not None and model is not None:
            found[TEMPLATE_KWARGS] = resolve(model, self.thinking)
        return found

    def passed_on(self) -> list[str]:
        """The flags every run gets unchanged."""
        argv: list[str] = []
        for flag, value in (
            ("--temperature", self.temperature),
            ("--top-p", self.top_p),
            ("--top-k", self.top_k),
        ):
            if value is not None:
                argv += [flag, str(value)]
        if self.stream:
            argv.append("--stream")
        if self.serving_note is not None:
            argv += ["--serving-note", str(self.serving_note)]
        if self.hardware_label is not None:
            argv += ["--hardware-label", self.hardware_label]
        if self.image is not None:
            argv += ["--image", self.image]
        if self.chat_template_kwargs is not None:
            argv += ["--chat-template-kwargs", self.chat_template_kwargs]
        if self.thinking is not None:
            argv += ["--thinking", self.thinking]
        return argv


@dataclass
class Boundaries:
    """Everything a start reaches outside its own code; tests replace each."""

    stdin: Any
    stdout: Any
    stderr: Any
    ports: tuple[int, ...] = discovery.USUAL_PORTS
    probe_seconds: float = discovery.PROBE_SECONDS
    readiness: Callable[[str], dict[str, Any]] | None = None
    # Called with a pass's command line; for a series whose server gets no
    # key while one is set, also with `env`, the environment without it.
    runner: Callable[..., child.Child] = child.run
    submitter: Callable[..., int] | None = None
    clock: Callable[[], float] = time.monotonic
    now: Callable[[], datetime] = datetime.now
    runs: Path = Path("runs")
    modes: dict[str, Mode] = field(default_factory=modes)
    osrelease: Path = OSRELEASE
    hosts: Path = HOSTS
    mac_files: Path = MAC_FILES


class Refused(Exception):
    """A step refused or failed; the text says what and, where known, how
    to fix it."""


def _names(items: list[str]) -> str:
    if len(items) < 2:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def choose_cap(
    given: int | None, context: int | None, case_ids: tuple[str, ...]
) -> tuple[int, str]:
    """The per-response cap of the passes and why: a given one; otherwise the
    largest cap where the reported context covers the largest measured need of
    `case_ids` with it, else the recommended minimum. `context_text` says that
    a model can need more than the measured need."""
    if given is not None:
        return given, "set with --max-tokens-per-call"
    largest = max(context_need(case_id, LARGEST_CAP) for case_id in case_ids)
    if context is None:
        return MIN_TOKENS_PER_CALL, (
            "chosen automatically: the recommended minimum, as the server reports"
            f" no context; --max-tokens-per-call {LARGEST_CAP} sets the largest cap"
        )
    if context >= largest:
        return LARGEST_CAP, (
            "chosen automatically: the largest cap, as the server's context of"
            f" {context:,} tokens covers this mode's measured need of {largest:,}"
            " with it"
        )
    return MIN_TOKENS_PER_CALL, (
        f"chosen automatically: the recommended minimum, as the server's context"
        f" of {context:,} tokens is below this mode's measured need of"
        f" {largest:,} with the largest cap of {LARGEST_CAP:,}"
    )


def context_text(mode: Mode, cap: int, context: int | None) -> str:
    """The context `mode` needs with `cap`, its measured need, and how the
    reported one fits; a covered need says that a model that carries more
    reasoning per step can need more."""
    recommended, short = context_recommendation(
        mode.case_ids, context, max_tokens_per_call=cap
    )
    largest = max(context_need(case_id, cap) for case_id in mode.case_ids)
    need = (
        f"needs up to {largest:,} tokens of context with this cap (recommended"
        f" {recommended:,})"
    )
    if context is None:
        return f"{need}; the server reports no context"
    if short:
        verb = "needs" if len(short) == 1 else "need"
        return (
            f"{need}; with the server's {context:,} tokens {_names(short)} {verb}"
            " more and may end with a capacity rejection, a technical result"
        )
    return (
        f"{need}; the server's {context:,} tokens cover that, though a model that"
        " carries more reasoning per step can need more"
    )


def hardware_lines(serving: dict[str, Any]) -> list[str]:
    """The hardware the server check read: the label and, for a loopback
    endpoint, this computer's GPUs, from the check's own read."""
    host = serving.get("host") or {}
    lines = []
    if host.get("label") is not None:
        lines.append(f"Hardware label: {shown(host['label'])}")
    accelerators = host.get("accelerators")
    reason = host.get("reason")
    if accelerators:
        found = [
            shown(item["name"])
            + ("" if item.get("memory_gib") is None else f" ({item['memory_gib']} GiB)")
            for item in accelerators
        ]
        lines.append(f"GPUs on this computer: {', '.join(found)}")
    elif reason == NOT_LOOPBACK:
        lines.append(
            "GPUs: not read, as the server is not on this computer's loopback"
            " address (--hardware-label states the hardware)"
        )
    elif reason is not None:
        lines.append(f"GPUs: not read ({reason})")
    return lines


# Why a pass's first attempt failed, as the result page records it: kind,
# status and reason.
Cause = tuple[str, str | None, str | None]
INTERRUPTED: Cause = ("interrupted", None, None)


def repeat_record(seed: int, cause: Cause) -> dict[str, Any]:
    """The result page's record of pass `seed`, repeated after `cause`."""
    kind, status, reason = cause
    return {
        "pass": seed,
        "seed": seed,
        "kind": kind,
        "status": status,
        "reason": reason,
    }


@dataclass
class Series:
    """A started series: where it lives, what it runs and what happened."""

    options: Options
    endpoint: str
    model: str
    mode: Mode
    cap: int
    folder: Path
    # The passes run without the API key although this process holds one: the
    # search found the server. A pass run again by hand must not send it either.
    keyless: bool = False
    passes: list[Path] = field(default_factory=list)
    repeats: list[str] = field(default_factory=list)
    # The repeats as the result page records them.
    repeat_records: list[dict[str, Any]] = field(default_factory=list)
    unmeasured: list[int] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)
    # The passes in the evaluation's order, once it is written.
    sources: list[Path] = field(default_factory=list)

    def run_argv(self, seed: int, output: Path) -> list[str]:
        """The command line of pass `seed`. It never holds `--instance`: the
        run draws its own instance value, as a repeat of the pass does."""
        return [
            "run",
            "--endpoint",
            self.endpoint,
            "--model",
            self.model,
            "--max-tokens-per-call",
            str(self.cap),
            "--seed",
            str(seed),
            *self.mode.selection,
            *self.options.passed_on(),
            "--output",
            str(output),
        ]

    def compare_argv(self, passes: list[Path]) -> list[str]:
        argv = ["compare", *map(str, passes), "--output", str(self.evaluation)]
        if self.mode.case_set is not None:
            argv += ["--set", self.mode.case_set]
        return argv

    @property
    def evaluation(self) -> Path:
        return self.folder / EVALUATION

    @property
    def result(self) -> Path:
        return self.folder / RESULT

    def attempt(self, seed: int) -> Path:
        """The next new folder for pass `seed`: `pass-k`, then its repeats."""
        first = self.folder / f"pass-{seed}"
        if not (first.exists() or first.is_symlink()):
            return first
        number = 1
        while True:
            repeat = self.folder / f"pass-{seed}-repeat-{number}"
            if not (repeat.exists() or repeat.is_symlink()):
                return repeat
            number += 1

    def commands(self, seed: int, cause: Cause) -> list[str]:
        """The commands that finish the series by hand from pass `seed` on,
        into new folders: the pass again, the passes after it, the
        evaluation and the result page. The page discloses every repeat
        already made and pass `seed`, repeated because its first attempt
        failed with `cause`, and marks a trial or a single pass as one. A
        pass of a series that runs without the API key is printed without it
        too: `werkbank run` would otherwise read the variable and send the key
        to the server the search found."""
        from werkbank_result.writer import repeat_text

        passes = list(self.passes)
        repeats = [*self.repeat_records, repeat_record(seed, cause)]
        lines = []
        for k in self.mode.seeds[self.mode.seeds.index(seed) :]:
            folder = self.attempt(k)
            passes.append(folder)
            unset = ["env", "-u", API_KEY_VARIABLE] if self.keyless else []
            lines.append(shlex.join([*unset, "werkbank", *self.run_argv(k, folder)]))
        lines.append(shlex.join(["werkbank", *self.compare_argv(passes)]))
        result = ["werkbank", "result", str(self.evaluation)]
        for repeat in repeats:
            result += ["--repeated", repeat_text(repeat)]
        if self.mode.trial:
            result.append("--trial")
        elif self.mode.single:
            result.append("--single-pass")
        lines.append(shlex.join([*result, "--output", str(self.result)]))
        return lines


def _readiness(
    console: Console, check: Callable[[str], dict[str, Any]], image: str, pinned: bool
) -> None:
    ready = check(image)
    if ready["ready"]:
        console.say("Docker, Harbor and the container image: ready")
        return
    fixes = FIXES if pinned else {**FIXES, **OTHER_IMAGE_FIXES}
    lines = ["Werkbank cannot run cases on this computer yet:"]
    for error in ready["errors"]:
        lines.append(f"- {error}")
        lines += [f"  Fix: {fix.format(image=image)}" for fix in fixes.get(error, ())]
    lines.append("Then check again with: werkbank harbor-doctor")
    raise Refused("\n".join(lines))


def _typed_endpoint(console: Console) -> str:
    """A base URL the tester types; Enter stops."""
    for _ in range(4):
        url = console.answer(
            "Base URL of your server (for example http://127.0.0.1:8000/v1),"
            " or Enter to stop: "
        ).strip()
        if not url:
            raise Stopped
        try:
            IsolatedEndpoint(url, discovery.PLACEHOLDER)
        except ValueError as error:
            # The transport's texts are fixed; any other refusal may echo it.
            why = str(error) if isinstance(error, EndpointError) else NOT_A_URL
            console.say(f"Not usable: {why}.")
            continue
        return url
    raise Stopped


def _lima(hosts: Path) -> bool:
    """Whether the hosts file names host.lima.internal as Lima writes it."""
    try:
        with hosts.open("rb") as file:
            text = file.read(HOSTS_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return False
    return any(LIMA_LINE.match(line) for line in text.splitlines())


def _host_hint(env: Boundaries) -> str:
    """Where a server outside this Linux is found, with a leading space: the
    WSL2 note under WSL2, the Lima note in a Lima VM, the OrbStack note in an
    OrbStack machine; else nothing. It reads local files only and sends no
    request."""
    try:
        release = env.osrelease.read_text(encoding="utf-8", errors="replace")
    except OSError:
        release = ""
    if "microsoft" in release.lower():
        return f" {WSL_NOTE}"
    if _lima(env.hosts):
        return f" {LIMA_NOTE}"
    if env.mac_files.is_dir():
        return f" {ORBSTACK_NOTE}"
    return ""


def _server(
    options: Options, env: Boundaries, console: Console
) -> tuple[str, discovery.Listing | None]:
    """The endpoint and, where discovery read it, its model list."""
    if options.endpoint is not None:
        return options.endpoint, None
    ports = ", ".join(str(port) for port in env.ports)
    console.say(
        f"Looking for a server on this computer: 127.0.0.1, ports {ports}"
        " (one read-only model list request each)"
    )
    found = discovery.discover(env.ports, timeout=env.probe_seconds)
    for url in found.locked:
        console.say(KEY_NEEDED.format(url=url))
    servers = found.servers

    def described(listing: discovery.Listing) -> str:
        port = int(listing.base_url.rsplit(":", 1)[1].split("/")[0])
        usual = discovery.usual_server(port)
        count = len(listing.models)
        text = f"{listing.base_url}, {count} model{'s' if count != 1 else ''}"
        return f"{text} (port usually used by {usual})" if usual else text

    if len(servers) == 1:
        console.say(f"Found a server: {described(servers[0])}")
        return servers[0].base_url, servers[0]
    if len(servers) > 1:
        if options.yes:
            raise Refused(
                "Several servers answered: "
                + "; ".join(listing.base_url for listing in servers)
                + ". Pass the one to use with --endpoint URL."
            )
        index = console.choose(
            "Several servers answered. Which one should Werkbank use?",
            [described(listing) for listing in servers],
        )
        return servers[index].base_url, servers[index]
    if options.yes:
        raise Refused(
            "No server answered on the usual ports. Pass its base URL with"
            f" --endpoint URL.{_host_hint(env)}"
        )
    console.say(f"No server answered on the usual ports.{_host_hint(env)}")
    return _typed_endpoint(console), None


def _model(
    options: Options,
    endpoint: str,
    listing: discovery.Listing | None,
    env: Boundaries,
    console: Console,
    api_key: str | None = None,
) -> str:
    if options.model is not None:
        return options.model
    if listing is None:
        try:
            # The fingerprint's deadline: this endpoint was named, not searched.
            listing = discovery.models(endpoint, timeout=PROBE_SECONDS, api_key=api_key)
        except discovery.NoList as error:
            if error.locked and api_key is not None:
                raise Refused(
                    "The server's model list could not be read: the server"
                    f" refused the API key in {API_KEY_VARIABLE}."
                ) from None
            if error.locked:
                raise Refused(
                    "The server's model list could not be read: the server needs"
                    f" a key. Set the environment variable {API_KEY_VARIABLE} to"
                    " it and start again."
                ) from None
            raise Refused(
                "The server's model list could not be read (no model list"
                " arrived); pass the served model name with --model NAME."
            ) from None
    if listing.paths:
        console.say(
            PATH_HINT.format(count=listing.paths, s="s" if listing.paths != 1 else "")
        )
    names = list(listing.models)
    if not names:
        raise Refused(
            "The server lists no model a run can use: load a model in the server,"
            " served under a name, and start again."
        )
    if len(names) == 1:
        console.say(f"Model: {shown(names[0])}")
        return names[0]
    if options.yes:
        raise Refused(
            f"The server lists {len(names)} models; pass the one to use with"
            " --model NAME."
        )
    if len(names) > MAX_NAMES:
        raise Refused(
            f"The server lists {len(names)} models; pass the one to use with"
            " --model NAME."
        )
    index = console.choose(
        "The server lists several models. Which one should Werkbank run?",
        [shown(name) for name in names],
    )
    return names[index]


def _check_server(
    options: Options,
    endpoint: str,
    model: str,
    console: Console,
    api_key: str | None = None,
) -> int | None:
    """The server check `werkbank preflight` makes, with the given cap or the
    recommended minimum; the context the server reports. `api_key` is the
    key of an endpoint the tester named, None for a server the search found."""
    from victors_werkbank.preflight import check, exit_code, render

    cap = options.max_tokens_per_call or MIN_TOKENS_PER_CALL
    console.say(
        "Checking the server: the fingerprint a run reads, then at most two short"
        f" chat requests with a cap of {cap:,} tokens"
    )
    try:
        sampling = options.sampling(model)
    except RecipeError as error:
        raise Refused(f"Cannot send --thinking to this model: {error}.") from None
    client = IsolatedEndpoint(
        endpoint, model, sampling, streaming=options.stream, api_key=api_key
    )
    result = check(
        client,
        max_tokens_per_call=cap,
        serving_note=options.note,
        hardware_label=options.hardware_label,
    )
    for line in render(result):
        # Start has exit codes of its own: the check's verdict without its.
        if line.startswith(VERDICT):
            line = line.split(" (exit ", 1)[0]
        console.say(line)
    for line in hardware_lines(result.serving):
        console.say(line)
    if exit_code(result) != 0:
        raise Refused(
            "The server is not ready for Werkbank: fix what the check names above"
            " (the hints say how), then start again."
        )
    return result.context_tokens


def _mode(
    options: Options,
    env: Boundaries,
    console: Console,
    context: int | None,
) -> tuple[Mode, int, str]:
    # The number of passes of a benchmark mode is asked after the mode unless
    # --passes gave it or --yes, which never asks, runs three.
    asked = options.passes is None and not options.yes
    found = [
        replace(mode, single=True) if options.passes == 1 and not mode.trial else mode
        for mode in env.modes.values()
    ]
    options_text = []
    for mode in found:
        cap, why = choose_cap(options.max_tokens_per_call, context, mode.case_ids)
        about = f"\n     {mode.about}" if mode.about else ""
        passes = mode.passes_text()
        if asked and not mode.trial:
            passes += f", or one pass with seed {SINGLE_SEEDS[0]}"
        options_text.append(
            f"{mode.title}: {mode.scope()}, {passes}{about}\n"
            f"     Per-response cap {cap:,}: {why}\n"
            f"     Context: {context_text(mode, cap, context)}"
        )
    if options.mode is not None:
        index = list(env.modes).index(options.mode)
        mode = found[index]
        console.say(f"Mode: {options_text[index]}")
    else:
        index = console.choose("What should Werkbank run?", options_text)
        mode = found[index]
    if asked and not mode.trial:
        if console.choose(PASSES_QUESTION, list(PASSES_TEXTS)) == 1:
            mode = replace(mode, single=True)
    cap, why = choose_cap(options.max_tokens_per_call, context, mode.case_ids)
    return mode, cap, why


def _folder(options: Options, env: Boundaries, mode: Mode) -> Path:
    """The new series folder, mode 0700; its name holds no model, host or
    endpoint."""
    try:
        if options.output is not None:
            try:
                options.output.mkdir(mode=0o700, parents=True)
            except FileExistsError:
                raise Refused(
                    "--output must be a new directory; nothing is overwritten"
                ) from None
            return options.output
        env.runs.mkdir(mode=0o700, parents=True, exist_ok=True)
        single = "-single" if mode.single else ""
        stem = f"start-{env.now().strftime('%Y%m%d-%H%M%S')}-{mode.name}{single}"
        for number in range(1, 101):
            folder = env.runs / (stem if number == 1 else f"{stem}-{number}")
            try:
                folder.mkdir(mode=0o700)
            except FileExistsError:
                continue
            return folder
    except OSError as error:
        raise Refused(
            f"The series folder cannot be created ({type(error).__name__})"
        ) from None
    raise Refused("No new series folder could be created under runs/")


@dataclass(frozen=True)
class Outcome:
    """A finished pass: why it failed technically (None when measured) in
    words and as the result page's fixed-form cause (kind, status, reason),
    its run's results and records, and a note on a deadline stop."""

    problem: str | None
    cause: Cause | None = None
    results: list[dict[str, Any]] | None = None
    view: Any = None
    note: str | None = None


def inspect(folder: Path) -> Outcome:
    """Classify a pass from its folder alone, never from its output: no
    folder, no `run.json`, records that do not validate or an `invalid`
    result is a technical failure; anything else, a deadline stop and every
    censored result included, is measured."""
    from victors_werkbank.results.records import stops_run
    from victors_werkbank.results.run import RecordError, read_run

    none = ("no_run_record", None, None)
    if folder.is_symlink() or not folder.is_dir():
        return Outcome("it did not start; werkbank run named the reason above", none)
    try:
        view = read_run(folder, complete=False)
    except RecordError as error:
        return Outcome(f"its records do not validate ({error})", none)
    except (OSError, ValueError) as error:
        return Outcome(f"its records cannot be read ({type(error).__name__})", none)
    if view.run is None:
        return Outcome(
            "it was interrupted before it finished (its folder holds no run.json)",
            ("interrupted", None, None),
        )
    results = view.run["results"]
    for result in results:
        if result["status"] == "invalid":
            return Outcome(
                f"{result['case_id']} ended invalid/{result['reason']}, a technical"
                " result that stops a run; the cases after it were not run",
                ("technical_result", "invalid", result["reason"]),
            )
    note = None
    for result in results:
        if stops_run(result):
            note = (
                f"{result['case_id']} reached its {result['reason'].replace('_', ' ')};"
                " the cases after it were not run. This counts as measured. The"
                " server may still be generating the unanswered request."
            )
    return Outcome(None, None, results, view, note)


def _complete(results: list[dict[str, Any]]) -> bool:
    from victors_werkbank.cli import _exit_code

    return _exit_code(results) == 0


SETUP_PARTS = (
    ("the served model", lambda plan: plan["model_id"]),
    (
        "the server's model listing (for example its context)",
        lambda plan: plan["serving"]["models"]["entry"],
    ),
    (
        "the engine's version or settings",
        lambda plan: {
            k: v
            for k, v in plan["serving"]["engine"].items()
            if k not in {"reason", "chat_template_defaults"}
        },
    ),
    (
        "the server's default template variables (such as its thinking default)",
        lambda plan: {
            k: v
            for k, v in (
                plan["serving"]["engine"].get("chat_template_defaults") or {}
            ).items()
            if k != "reason"
        },
    ),
    (
        "the hardware (label or GPUs read)",
        lambda plan: {
            k: plan["serving"]["host"].get(k) for k in ("label", "accelerators")
        },
    ),
    ("the serving note", lambda plan: plan["serving"]["owner_note"]),
    (
        "the run settings (sampling, thinking, cap, streaming)",
        lambda plan: plan["serving"]["configuration"],
    ),
    ("the container image", lambda plan: plan["image"].rsplit("@", 1)[-1]),
)


def _same_setup(first: Any, later: Any, number: int) -> None:
    """Refused where pass `number` was run with another setup than pass 1,
    so its runs would not pool."""
    from victors_werkbank.results.evaluation import pool_key

    if pool_key(first) == pool_key(later):
        return
    changed = [
        name for name, part in SETUP_PARTS if part(first.plan) != part(later.plan)
    ]
    what = _names(changed) if changed else "the recorded setup"
    raise Refused(
        f"The server setup changed between pass 1 and pass {number}: {what}."
        " Scores need three passes of one unchanged setup; start a new series"
        " once the server runs unchanged."
    )


def child_environment(api_key: str | None) -> dict[str, str] | None:
    """The environment of a series' `werkbank run` children, which read
    `WERKBANK_API_KEY` themselves: None, this process's own, where it already
    gives them exactly `api_key` (the key of an endpoint the tester named, or
    none); otherwise this process's with the variable removed, for a server
    the search found, or set to `api_key`."""
    if (os.environ.get(API_KEY_VARIABLE) or None) == api_key:
        return None
    found = {k: v for k, v in os.environ.items() if k != API_KEY_VARIABLE}
    if api_key is not None:
        found[API_KEY_VARIABLE] = api_key
    return found


def _passes(
    series: Series,
    env: Boundaries,
    console: Console,
    environment: dict[str, str] | None = None,
) -> int | None:
    """Run the passes; None when all are measured, else the exit code. With
    `environment`, every pass runs in it instead of this process's own."""
    episodes = series.mode.episodes()
    seeds = series.mode.seeds
    first_view = None
    for number, seed in enumerate(seeds, start=1):
        if series.seconds:
            pace = sum(series.seconds) / len(series.seconds)
            left = len(seeds) - number + 1
            rest = (
                "the last pass takes"
                if left == 1
                else f"the remaining {left} passes take"
            )
            console.say(
                f"Measured pace: {duration(pace)} per pass; {rest} about"
                f" {duration(pace * left)}."
            )
        repeated = None
        cause: Cause | None = None
        while True:
            folder = series.attempt(seed)
            console.step("pass", seed=seed, folder=folder, episodes=episodes)
            console.say(
                f"Pass {seed} of {len(seeds)} (seed {seed}): {episodes}"
                f" episode{'s' if episodes != 1 else ''} into {folder}"
            )
            started = env.clock()
            try:
                argv = series.run_argv(seed, folder)
                if environment is None:
                    done = env.runner(argv)
                else:
                    done = env.runner(argv, env=environment)
            except OSError as error:
                raise Refused(
                    f"werkbank run could not be started ({type(error).__name__})"
                ) from None
            took = env.clock() - started
            if done.interrupted:
                console.error(
                    f"Pass {seed} was stopped; its folder {folder} stays as"
                    " evidence. To finish the series by hand, once the server is"
                    " idle:"
                )
                for line in series.commands(seed, cause or INTERRUPTED):
                    console.error(f"  {line}")
                return STOPPED
            outcome = inspect(folder)
            if outcome.problem is None:
                break
            console.error(
                f"Pass {seed} failed technically: {outcome.problem}. Its folder"
                f" {folder} stays as evidence. Repeat the pass once the server is"
                " idle; nothing was retried."
            )
            if (
                repeated is not None
                or series.options.yes
                or not console.confirm(f"Repeat pass {seed} now? [y/N] ", default=False)
            ):
                console.error("To finish the series by hand, once the server is idle:")
                assert outcome.cause is not None
                for line in series.commands(seed, cause or outcome.cause):
                    console.error(f"  {line}")
                return FAILED
            repeated = (
                f"pass {seed} was repeated: {folder.name} failed technically,"
                f" {outcome.problem}"
            )
            cause = outcome.cause
        series.seconds.append(took)
        console.say(f"Pass {seed} took {duration(took)}.")
        if outcome.note is not None:
            console.say(outcome.note)
        if repeated is not None and cause is not None:
            series.repeats.append(f"{repeated}; only {folder.name} is evaluated")
            series.repeat_records.append(repeat_record(seed, cause))
        assert outcome.results is not None
        if not _complete(outcome.results):
            series.unmeasured.append(seed)
        if first_view is None:
            first_view = outcome.view
        else:
            _same_setup(first_view, outcome.view, seed)
        series.passes.append(folder)
        series.run_ids.append(outcome.view.plan["run_id"])
    return None


def _evaluate(series: Series, console: Console) -> dict[str, Any]:
    """`werkbank compare` of the passes and the result page beside it."""
    from victors_werkbank.results.evaluation import write_evaluation
    from victors_werkbank.results.run import RecordError
    from werkbank_result import ResultError, write

    passes = "the trial pass" if series.mode.trial else "the three passes"
    if series.mode.single:
        passes = "the one pass"
    console.step("evaluation")
    console.say(f"Evaluating {passes} into {series.evaluation}")
    try:
        record = write_evaluation(
            series.passes, series.evaluation, case_set=series.mode.case_set
        )
    except RecordError as error:
        raise Refused(f"Cannot compare: {error}") from None
    except (ValueError, OSError) as error:
        raise Refused(f"Cannot compare: {type(error).__name__}") from None
    folders = dict(zip(series.run_ids, series.passes, strict=True))
    series.sources = [folders[item["run_id"]] for item in record["inputs"]]
    try:
        write(
            series.evaluation,
            series.result,
            repeats=series.repeat_records,
            trial=series.mode.trial,
            single_pass=series.mode.single,
        )
    except ResultError as error:
        raise Refused(f"Cannot write the result page: {error}") from None
    except (ValueError, OSError) as error:
        raise Refused(f"Cannot write the result page: {type(error).__name__}") from None
    return record


def _solved_line(
    record: dict[str, Any], run_ids: list[str], seeds: tuple[int, ...] = SEEDS
) -> str | None:
    """The tasks each pass solved, by pass: the evaluation lists its runs by
    run ID, so each count is placed by the run ID of pass k (`run_ids`)."""
    from victors_werkbank.results.outcomes import per_pass
    from victors_werkbank.results.run import RecordError

    if len(record["pools"]) != 1:
        return None
    pool = record["pools"][0]
    try:
        found = per_pass(pool)
    except RecordError:
        return None
    if found is None:
        return "Tasks solved per pass: not evaluated (see the result page)"
    solved = dict(zip(pool["runs"], found["passes"], strict=True))
    if sorted(solved) != sorted(run_ids):
        return None
    passes = ", ".join(
        f"pass {seed}: {solved[run_id]}"
        for seed, run_id in zip(seeds, run_ids, strict=True)
    )
    return (
        f"Tasks solved per pass, of {found['tasks']}: {passes}"
        f" (on average {found['mean']:.1f})"
    )


def _fulfilment_lines(record: dict[str, Any]) -> list[str]:
    """Beside the tasks solved, the subgoal fulfilment overall and per
    category beside its solved attempts, as `werkbank compare` prints it,
    and what it means; nothing for a pool that is not evaluated."""
    from victors_werkbank.results.outcomes import FULFILMENT_NOTE, fulfilment_lines
    from victors_werkbank.results.run import RecordError

    if len(record["pools"]) != 1:
        return []
    try:
        lines = fulfilment_lines(record["pools"][0])
    except RecordError:
        return []
    return [*lines, FULFILMENT_NOTE] if lines else []


def _summary(
    series: Series, record: dict[str, Any], why: str, context: int | None
) -> list[str]:
    lines = [
        "",
        f"Series written: {series.folder}",
        "Passes evaluated: "
        + ", ".join(
            f"{folder.name} (seed {seed})"
            for seed, folder in zip(series.mode.seeds, series.passes, strict=True)
        ),
    ]
    lines += [f"Repeated: {text}" for text in series.repeats]
    if series.mode.trial:
        lines += _trial_lines(record)
    else:
        solved = _solved_line(record, series.run_ids, series.mode.seeds)
        if solved is not None:
            lines.append(solved)
        lines += _fulfilment_lines(record)
        if series.mode.single:
            lines.append(SINGLE_SUMMARY)
    if series.unmeasured:
        lines.append(
            "Not every case was measured: "
            + _names([f"pass {seed}" for seed in series.unmeasured])
            + " left cases unmeasured; the evaluation shows bounds"
        )
    lines += [
        f"Per-response cap {series.cap:,}: {why}",
        f"Context: this mode {context_text(series.mode, series.cap, context)}",
        f"Evaluation: {series.evaluation}",
        f"Result page: {series.result / 'index.html'}",
    ]
    return lines


TRIAL_SUMMARY = (
    "This was a trial, not a benchmark: it has no score, compares with no other"
    " result and is never submitted. For results to compare, start Quick,"
    " Normal or Full."
)


def _trial_lines(record: dict[str, Any]) -> list[str]:
    """What the trial pass's attempts came to, by outcome group, and that it
    is no benchmark."""
    from victors_werkbank.results.outcomes import GROUP_TEXTS, group_counts

    lines = []
    if len(record["pools"]) == 1:
        counts = group_counts(record["pools"][0]["failures"]["overall"]["classes"])
        total = sum(counts.values())
        found = ", ".join(
            f"{GROUP_TEXTS[group]} {count}" for group, count in counts.items()
        )
        lines.append(f"Trial pass, {total} attempts: {found}")
    return [*lines, TRIAL_SUMMARY]


def _submit_command(series: Series) -> str:
    argv = [
        "werkbank",
        "submit",
        str(series.evaluation),
        "--result",
        str(series.result),
    ]
    for folder in series.sources:
        argv += ["--source", str(folder)]
    return shlex.join(argv)


def _offer(series: Series, env: Boundaries, console: Console) -> None:
    """The offer to submit: only at a terminal and never with --yes, which
    prints the command instead. On `y` the submit flow runs on the same
    streams with the passes as its sources and the endpoint for its privacy
    check; it shows exactly what is sent and asks again. Without a receiver
    it offers to write the archive into the series folder. Its outcome never
    changes the exit code: end of input, Ctrl-C or answers that are no y or n
    at this question count as no, as the series is already complete."""
    later = _submit_command(series)
    if series.options.yes or not console.terminal():
        console.say(
            "To send these results to the Werkbank maintainer, run at a terminal:"
            f" {later}"
        )
        return
    try:
        wanted = console.confirm(
            "Send these results to the Werkbank maintainer? You will see exactly"
            " what is sent and confirm again. [y/N] ",
            default=False,
        )
    except (Stopped, KeyboardInterrupt):
        wanted = False
    if not wanted:
        console.say(f"Nothing was sent. To send them later: {later}")
        return
    if env.submitter is None:
        from werkbank_submit import flow as submit_flow

        send = submit_flow.submit
    else:
        send = env.submitter
    console.step("submission")
    try:
        send(
            series.evaluation,
            series.result,
            list(series.sources),
            endpoint=series.endpoint,
            console=console.submission(),
        )
    except KeyboardInterrupt:
        console.say(f"The submission was stopped. To send them later: {later}")


def start(options: Options, env: Boundaries, console: Console | None = None) -> int:
    """The start on `console`, by default the terminal of `env`'s streams."""
    if console is None:
        console = Console(env.stdin, env.stdout, env.stderr)
    try:
        return _start(options, env, console)
    except Refused as error:
        console.error(str(error))
        return FAILED
    except (Stopped, KeyboardInterrupt):
        console.error("Stopped by you; nothing more was started.")
        return STOPPED


def _start(options: Options, env: Boundaries, console: Console) -> int:
    from victors_werkbank.harbor_smoke import SMOKE_IMAGE, preflight

    image = options.image or SMOKE_IMAGE
    console.step("readiness")
    _readiness(console, env.readiness or preflight, image, image == SMOKE_IMAGE)
    console.step("server")
    endpoint, listing = _server(options, env, console)
    # A key goes only to an endpoint the tester named, whose model list the
    # search did not read; a server the search found never gets it.
    api_key = options.api_key if listing is None else None
    if api_key is not None:
        console.say(KEY_SENT)
    elif options.api_key is not None:
        console.say(KEY_NOT_SENT)
    console.step("model")
    model = _model(options, endpoint, listing, env, console, api_key)
    console.step("check")
    context = _check_server(options, endpoint, model, console, api_key)
    console.step("mode")
    mode, cap, why = _mode(options, env, console, context)
    if not options.yes and not console.confirm(
        f"Start {mode.title} now: {mode.count_text()} with a cap of {cap:,}? [Y/n] ",
        default=True,
    ):
        raise Stopped
    folder = _folder(options, env, mode)
    series = Series(
        options,
        endpoint,
        model,
        mode,
        cap,
        folder,
        keyless=api_key is None and options.api_key is not None,
    )
    console.step("series", folder=folder, mode=mode)
    console.say(f"Series folder: {folder}")
    console.say(DRAWN_DATA)
    stopped = _passes(series, env, console, child_environment(api_key))
    if stopped is not None:
        return stopped
    record = _evaluate(series, console)
    console.step("result", evaluation=series.evaluation, result=series.result)
    for line in _summary(series, record, why, context):
        console.say(line)
    # A trial and a single pass are never submitted: no offer and no command
    # to submit them.
    if mode.submitted:
        _offer(series, env, console)
    return UNMEASURED if series.unmeasured else COMPLETE
