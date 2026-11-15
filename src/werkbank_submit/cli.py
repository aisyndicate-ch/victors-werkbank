"""`werkbank submit`: show an evaluation's submission, ask, then send it once
to the Werkbank maintainer or write it to a new local file."""

import argparse
from pathlib import Path

from . import flow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="werkbank submit",
        description=(
            "Submit an evaluation's results: asks five optional questions,"
            " checks the evaluation, its result page and, with --source, the"
            " evaluated runs, shows exactly what would be sent and the consent"
            " text, and sends only after an explicit 'y' at a terminal."
            " Exit 0: receipt confirmed or file written; 2: refused, something"
            " needs fixing; 3: nothing confirmed."
        ),
    )
    parser.add_argument(
        "evaluation", type=Path, metavar="EVALUATION", help="an evaluation directory"
    )
    parser.add_argument(
        "--result",
        type=Path,
        required=True,
        metavar="DIR",
        help="the evaluation's result page directory, beside the evaluation",
    )
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        default=[],
        metavar="RUN",
        help="an evaluated run, in any order (repeat for each); needed for a"
        " complete check and to offer raw data",
    )
    parser.add_argument(
        "--write",
        type=Path,
        metavar="FILE",
        help="write the archive to this new file instead of sending it; nothing"
        " is sent then",
    )
    args = parser.parse_args(argv)
    return flow.submit(args.evaluation, args.result, args.source, write=args.write)
