"""`werkbank result EVALUATION --output NEW [--repeated REPEAT ...]
[--trial | --single-pass]`: write an evaluation's result page offline,
disclosing the passes that were repeated and marking a trial or a single
pass. Exit 0: written, one JSON line naming the page; 2: refused, with a
fixed message that names no path, no argument's text and no file's
contents."""

import argparse
import json
import sys
from pathlib import Path

from .writer import PAGE, REPEAT_FORM, RESULT, ResultError, repeat_argument, write


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="werkbank result",
        description=(
            "Write the result page of an evaluation that this build wrote with"
            " `werkbank compare`: index.html and result.json in a new directory"
            " beside the evaluation, never inside it. Offline; no ranking."
        ),
    )
    parser.add_argument(
        "evaluation", type=Path, metavar="EVALUATION", help="evaluation directory"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="new result directory"
    )
    parser.add_argument(
        "--repeated",
        action="append",
        default=[],
        metavar=REPEAT_FORM,
        help=(
            "a pass that was repeated, disclosed on the page (repeatable): KIND"
            " is technical_result with invalid/REASON, interrupted or"
            " no_run_record; werkbank start prints it for a series finished by"
            " hand"
        ),
    )
    parser.add_argument(
        "--trial",
        action="store_true",
        help=(
            "mark the page as a trial of werkbank start, no benchmark: no score,"
            " and werkbank submit refuses it; the evaluation must be one pass"
        ),
    )
    parser.add_argument(
        "--single-pass",
        action="store_true",
        help=(
            "mark the page as a single pass of werkbank start's Quick, Normal or"
            " Full:"
            " the numbers stay, and werkbank submit refuses it; the evaluation"
            " must be one pass; not with --trial"
        ),
    )
    args = parser.parse_args(argv)
    label = "Cannot write the result page"
    try:
        repeats = [repeat_argument(text) for text in args.repeated]
        record = write(
            args.evaluation,
            args.output,
            repeats=repeats,
            trial=args.trial,
            single_pass=args.single_pass,
        )
    except ResultError as error:
        parser.exit(2, f"{label}: {error}\n")
    except (ValueError, OSError) as error:
        parser.exit(2, f"{label}: {type(error).__name__}\n")
    print(
        json.dumps(
            {
                "page": str(args.output / PAGE),
                "result": str(args.output / RESULT),
                "evaluation_sha256": record["evaluation"]["sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
