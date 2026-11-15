"""`werkbank start`: the guided path from a server on this computer to a
result page. Every argument is refused as `werkbank run` refuses it, before
any contact and before any output exists."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

from victors_werkbank.cli import (
    _add_thinking_arguments,
    _hardware_label,
    _integer,
    _number,
    _output_cap,
    _served_model,
)

from . import discovery
from .flow import FAILED, Boundaries, Options, start

DESCRIPTION = (
    "Guided start: checks that Docker, Harbor and the image are ready; finds"
    " a server on this computer (one read-only GET /v1/models on 127.0.0.1"
    " per usual port: "
    + ", ".join(str(port) for port in discovery.USUAL_PORTS)
    + ") unless --endpoint is given; picks the model; checks the server as"
    " werkbank preflight does; asks for the mode and, for Quick, Normal and"
    " Full, the number of passes; runs three passes with seeds 1, 2 and 3 (each a"
    " werkbank run), or one pass with seed 1 on request and for the trial;"
    " evaluates them (werkbank compare) and writes the result page"
    " into a new series folder under runs/. A request to the model goes out"
    " only after your choice, with exactly one server and model found, or"
    " with the flags. An API key in the environment variable"
    " WERKBANK_API_KEY is sent only to a server you name (--endpoint or a"
    " typed base URL), never by the search or to a server it found. At the"
    " end it offers to submit the results at a"
    " terminal; --yes never submits, and a trial or a single pass is never"
    " offered. Exit 0:"
    " every pass measured completely and the page written; 3: evaluated, but"
    " a pass left cases unmeasured; 2: refused or failed; 1: stopped by you."
)


class _Refusing(argparse.ArgumentParser):
    """A parser that raises ValueError with argparse's message instead of
    printing it and exiting."""

    def error(self, message: str) -> NoReturn:
        raise ValueError(message)


# The flags a form of the browser interface can set, by its field names; a
# checked `stream` field stands for --stream.
FORM_FLAGS = {
    "endpoint": "--endpoint",
    "model": "--model",
    "hardware_label": "--hardware-label",
    "max_tokens_per_call": "--max-tokens-per-call",
    "temperature": "--temperature",
    "top_p": "--top-p",
    "top_k": "--top-k",
    "thinking": "--thinking",
}
# What a form never sets and so keeps from the command line.
KEPT = ("mode", "passes", "output", "serving_note", "image", "chat_template_kwargs")


def form_options(
    defaults: argparse.Namespace | None, fields: dict[str, str]
) -> Options:
    """The checked options of a start begun from the browser interface's
    form: each filled field as the flag it stands for, through this
    command's own parser and checks, with what the form never sets taken
    from `defaults`, the command line the interface was started with.
    ValueError, with the parser's or the check's text, where the command
    line would be refused."""
    unknown = sorted(set(fields) - set(FORM_FLAGS) - {"stream"})
    if unknown:
        raise ValueError("unknown form field")
    argv = []
    for name, flag in FORM_FLAGS.items():
        value = fields.get(name, "").strip()
        if value:
            argv.append(f"{flag}={value}")
    if fields.get("stream"):
        argv.append("--stream")
    args = _parser(_Refusing).parse_args(argv)
    if defaults is not None:
        for name in KEPT:
            setattr(args, name, getattr(defaults, name))
    if args.thinking is not None and args.chat_template_kwargs is not None:
        raise ValueError("--thinking and --chat-template-kwargs exclude each other")
    return options(args)


def _parser(
    kind: type[argparse.ArgumentParser] = argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser = kind(prog="werkbank start", description=DESCRIPTION)
    parser.add_argument(
        "--endpoint", help="the server's HTTP(S) /v1 base URL; skips the search"
    )
    parser.add_argument(
        "--model", type=_served_model, help="the served model name to run"
    )
    parser.add_argument(
        "--mode",
        choices=("quick", "normal", "full", "trial"),
        help="quick: the case set quick-v3, the easy and medium cases that a"
        " 64K context can run; normal: the case set normal-v1, every case that is"
        " not very hard; full: the case set full-v1, every case; normal and full"
        " have an overall score; trial: one pass over seven easy cases that checks the installation,"
        " the agent loop, the containers and the result page (no benchmark,"
        " no score, never submitted)",
    )
    parser.add_argument(
        "--passes",
        type=int,
        choices=(1, 3),
        help="for quick, normal and full, 3: three passes with seeds 1, 2 and 3; 1: one"
        " pass with seed 1, about a third of the time, never submitted and"
        " without temperature 0 without a score. Without it start asks, and"
        " with --yes it runs three; not with --mode trial, which is one pass",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="never ask: anything ambiguous is refused with the flag to pass,"
        " a technically failed pass is not repeated, and nothing is ever"
        " submitted (the submit command is printed, except for a trial or a"
        " single pass); needs --mode",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="run the guided start in the browser interface instead of this"
        " terminal: it prints a local address (127.0.0.1 only) to open, the"
        " other flags become the form's defaults, and every question is"
        " answered there (docs/users/INTERFACE.md); not with --yes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="DIR",
        help="a new series folder (default: runs/start-<date>-<time>-<mode>)",
    )
    parser.add_argument(
        "--max-tokens-per-call",
        type=_output_cap,
        help="the per-response cap (1..65536); default: 65,536 where the"
        " server's context covers the mode with it, else 32,768",
    )
    sampling = parser.add_argument_group(
        "passed on unchanged",
        "as werkbank run takes them: to the server check and every pass, --image"
        " to the readiness check and every pass",
    )
    sampling.add_argument("--temperature", type=_number, help="0..2")
    sampling.add_argument("--top-p", type=_number, help="0 < top_p <= 1")
    sampling.add_argument("--top-k", type=_integer, help="1..1000000")
    sampling.add_argument("--stream", action="store_true", help="stream responses")
    sampling.add_argument(
        "--serving-note", type=Path, metavar="FILE", help="a serving note"
    )
    sampling.add_argument(
        "--hardware-label",
        type=_hardware_label,
        metavar="TEXT",
        help="the serving hardware in your words",
    )
    sampling.add_argument(
        "--image", metavar="IMAGE", help="the pinned container image of the passes"
    )
    # Passed on unchanged as well; nothing is sent unless one of them is given.
    _add_thinking_arguments(parser)
    return parser


def options(args: argparse.Namespace) -> Options:
    """The checked options; ValueError with a fixed text naming no value."""
    from victors_werkbank.agent.measurements import template_json, validate_sampling
    from victors_werkbank.agent.model_process import IsolatedEndpoint
    from victors_werkbank.agent.transport import (
        EndpointError,
        api_key_from_environment,
    )
    from victors_werkbank.harbor_smoke import HarborError, image_reference
    from victors_werkbank.serving import load_note
    from victors_werkbank.thinking import resolve

    if args.yes and args.mode is None:
        raise ValueError(
            "--yes needs --mode quick, --mode normal, --mode full or --mode trial"
        )
    if args.passes is not None and args.mode == "trial":
        raise ValueError(
            "--passes applies to quick, normal and full; the trial is one pass"
        )
    if args.yes and args.browser:
        raise ValueError("--browser asks its questions in the browser; not with --yes")
    try:
        # Kept in the environment: each pass reads the variable itself.
        api_key = api_key_from_environment(keep=True)
    except EndpointError as error:
        # A fixed text that never holds the value.
        raise ValueError(str(error)) from None
    found = Options(
        endpoint=args.endpoint,
        model=args.model,
        mode=args.mode,
        yes=args.yes,
        passes=args.passes,
        output=args.output,
        max_tokens_per_call=args.max_tokens_per_call,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stream=args.stream,
        serving_note=args.serving_note,
        hardware_label=args.hardware_label,
        image=args.image,
        chat_template_kwargs=None
        if args.chat_template_kwargs is None
        else template_json(args.chat_template_kwargs),
        thinking=args.thinking,
        api_key=api_key,
    )
    if args.thinking is not None and args.model is not None:
        # Refused before any contact; without --model it is resolved once the
        # model is chosen, before the server check sends a request.
        resolve(args.model, args.thinking)
    try:
        validate_sampling(found.sampling())
    except ValueError:
        raise ValueError("invalid sampling settings") from None
    if args.image is not None:
        try:
            image_reference(args.image)
        except HarborError as error:
            raise ValueError(str(error)) from None
    if args.endpoint is not None:
        try:
            IsolatedEndpoint(
                args.endpoint,
                args.model or discovery.PLACEHOLDER,
                found.sampling(),
                streaming=args.stream,
                api_key=api_key,
            )
        except ValueError as error:
            # The transport's texts are fixed and never hold the URL; any other
            # refusal is named by its type only, as run and preflight do.
            text = (
                str(error) if isinstance(error, EndpointError) else type(error).__name__
            )
            raise ValueError(text) from None
    if args.output is not None and (args.output.exists() or args.output.is_symlink()):
        raise ValueError("--output must be a new directory")
    note = None
    if args.serving_note is not None:
        # Names the note's keys and their rules, never a value or the path.
        note = load_note(args.serving_note)
    return replace(found, note=note)


def main(argv: list[str] | None = None, boundaries: Boundaries | None = None) -> int:
    env = boundaries or Boundaries(sys.stdin, sys.stdout, sys.stderr)
    try:
        args = _parser().parse_args(argv)
    except SystemExit as error:
        return error.code if isinstance(error.code, int) else FAILED
    try:
        checked = options(args)
    except ValueError as error:
        print(f"Cannot start: {error}", file=env.stderr, flush=True)
        return FAILED
    if args.browser:
        from werkbank_ui.cli import serve

        return serve([env.runs], defaults=args)
    return start(checked, env)


if __name__ == "__main__":
    sys.exit(main())
