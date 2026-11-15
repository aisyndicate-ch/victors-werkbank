"""`werkbank-ui`: serve the browser interface on loopback until interrupted:
the guided start and its series, and the view of run and evaluation
directories; with `--read-only`, the view alone."""

import argparse
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import Any

from .catalog import MARKERS
from .control import Control
from .server import Viewer

RUNS = Path("runs")


def serve(
    roots: list[Path],
    *,
    port: int = 0,
    read_only: bool = False,
    defaults: Any = None,
) -> int:
    """Serve `roots` until interrupted. Unless `read_only`, new series go
    under the first root, which is created where it is missing, and
    `defaults` are the `werkbank start` arguments the form begins with.
    Exit 0 after an interrupt, 2 where it cannot serve."""
    control = None
    if not read_only:
        runs = roots[0]
        try:
            runs.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as error:
            print(
                f"werkbank-ui: cannot create the series folder ({type(error).__name__})",
                file=sys.stderr,
            )
            return 2
        if any((runs / name).exists() for name, _ in MARKERS):
            print(
                "werkbank-ui: new series need a folder to land in, and the first"
                " directory is itself a run or an evaluation; name a folder that"
                " holds them first, or pass --read-only",
                file=sys.stderr,
            )
            return 2
        control = Control(runs, defaults)
    resolved = []
    for root in roots:
        if not root.is_dir():
            print(f"werkbank-ui: not a directory: {root}", file=sys.stderr)
            return 2
        resolved.append(root.resolve())
    try:
        server = Viewer(resolved, port, control)
    except OSError as error:
        print(f"werkbank-ui: cannot listen: {error.strerror}", file=sys.stderr)
        return 2
    count = f"{len(roots)} director{'y' if len(roots) == 1 else 'ies'}"
    what = (
        f"read-only view of {count}"
        if read_only
        else f"guided start and view of {count}; new series go under {roots[0]}"
    )
    print(
        f"werkbank-ui: {what}, at\n  {server.url}\n"
        "Open it in a browser on this machine, under WSL2 also on the Windows"
        " side; stop with Ctrl-C."
        + (
            ""
            if read_only
            else " A series runs in this process: keep it running until the"
            " series has ended; stopping it stops the running pass."
        ),
        flush=True,
    )

    def ended(signum: int, frame: FrameType | None) -> None:
        raise KeyboardInterrupt

    # A hang-up or a termination ends the interface as Ctrl-C does, so a
    # running pass is stopped by its own rules and never left behind.
    previous = {
        number: signal.signal(number, ended)
        for number in (signal.SIGTERM, signal.SIGHUP)
    }
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if control is not None and control.session is not None:
            if control.session.running():
                print(
                    "werkbank-ui: stopping the running series; its pass ends by"
                    " its own rules, which can take a minute",
                    file=sys.stderr,
                    flush=True,
                )
            _until_stopped(control)
        server.server_close()
        for number, handler in previous.items():
            signal.signal(number, handler)
    return 0


def _until_stopped(control: Control) -> None:
    """Wait for the stopped series; a further interrupt does not cut the
    wait short, as the pass must end first."""
    while True:
        try:
            control.shutdown()
            return
        except KeyboardInterrupt:
            continue


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="werkbank-ui",
        description=(
            "The browser interface: the guided start of werkbank start (server,"
            " model, check, mode, passes, evaluation, result page, offer to"
            " submit) and a view of run and evaluation directories. Listens on"
            " 127.0.0.1 only; open the printed address, which carries this"
            " session's token. With --read-only it starts and writes nothing."
            " To give the start's flags as defaults, begin it as werkbank"
            " start --browser."
        ),
    )
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        metavar="DIRECTORY",
        help="a run or evaluation directory, or a directory holding them"
        " (default: runs); new series go under the first one",
    )
    parser.add_argument(
        "--port", type=int, default=0, help="loopback port (default: a free one)"
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="only the view: nothing is started and nothing is written",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be 0 to 65535")
    if args.read_only and not args.roots:
        parser.error("--read-only needs at least one DIRECTORY")
    return serve(args.roots or [RUNS], port=args.port, read_only=args.read_only)


if __name__ == "__main__":
    sys.exit(main())
