"""The fixed texts of a submission: format identifiers, the questionnaire, the
consent text, the prompts and the messages for each receiver answer, exactly
as `docs/SUBMISSION.md` states them (a test holds the two together)."""

SUBMISSION_FORMAT = "werkbank-submission-v1"
QUESTIONS_FORMAT = "werkbank-questions-v1"
CONSENT_FORMAT = "werkbank-consent-v2"
RECEIPT_FORMAT = "werkbank-receipt-v1"

# (field, question, ((choice ID, label), ...)); every question can be skipped.
QUESTIONS: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    (
        "purpose",
        "What did you run Werkbank for?",
        (
            ("choose_model", "Choosing a model"),
            (
                "compare_setup",
                "Comparing serving setups (engine, quantization, hardware)",
            ),
            ("check_server", "Checking my server"),
            ("curiosity", "Curiosity"),
            ("other", "Something else"),
        ),
    ),
    (
        "setup",
        "How did installing and starting go?",
        (
            ("smooth", "Smoothly"),
            ("minor_problems", "Minor problems"),
            ("major_problems", "Major problems"),
        ),
    ),
    (
        "duration",
        "How was the run time?",
        (
            ("fine", "Fine"),
            ("long_but_ok", "Long but acceptable"),
            ("too_long", "Too long"),
        ),
    ),
    (
        "clarity",
        "How clear is the result page?",
        (
            ("clear", "Clear"),
            ("partly", "Partly clear"),
            ("unclear", "Unclear"),
        ),
    ),
    (
        "plausibility",
        "Do the results match your own experience with this model?",
        (
            ("yes", "Yes"),
            ("partly", "Partly"),
            ("no", "No"),
            ("no_experience", "I have no experience with it"),
        ),
    ),
)

CONSENT_TEXT = """\
This sends your results to the Werkbank maintainer at the address shown above.

Always sent:
- the result page and the evaluation tables of this comparison
- the Werkbank version and its build digest, and the case set
- the served model name, the serving settings Werkbank recorded, the
  hardware statement (your hardware label as you wrote it, or the
  detected GPU product names) and the container image digest, listed
  with their values in the summary above
- your answers to the questions (skipped ones stay empty)
- a random installation ID, stored on this computer, that identifies
  your submissions without an account

Never sent: your endpoint address, host names, user names, file paths,
API keys or other credentials.

Raw data is sent only if you answer yes to "Include raw data?". It is
each run's export archive: every model response including its
reasoning, every tool call and result, the task files, the run records
and the server's model listing, in which a local model location appears
only as its last part and a hash.

The maintainer uses submissions to improve Werkbank and its cases. A
submission is your own report: it is never ranked and never published
without your agreement. It is kept until you ask for deletion, at most
12 months after it arrives. To have every submission under your
installation ID deleted, write to the maintainer at the contact shown
above and quote the ID. A deleted submission is gone from the
receiver's backups within 30 days.

The receiver sees your IP address, as every web server does. It never
stores it with your submission; its server logs keep it for at most
7 days, only to limit abuse.
"""

INCLUDE_RAW = "Include raw data? [y/N] "
SEND_NOW = "Send now? [y/N] "
WRITE_FILE = "Write the file? [y/N] "
CHOICE = "Choice (1-{count}, Enter to skip): "
CHOICE_AGAIN = "Enter a number from 1 to {count}, or press Enter to skip."

# What the client shows for each receiver answer.
RECEIVED = "Received, receipt {receipt}."
DUPLICATE = "Already received; nothing new was stored."
MALFORMED = "Refused as malformed; nothing was stored. Please report this."
TOO_LARGE = "Too large for the receiver; nothing was stored."
WITHOUT_RAW = "Send it again without raw data."
UNSUPPORTED = "The receiver does not accept this format; update Werkbank."
RATE_LIMITED = "Too many submissions; try again in {minutes} minutes."
RATE_LIMITED_ONE = "Too many submissions; try again in 1 minute."
RATE_LIMITED_LATER = "Too many submissions; try again later."
UNAVAILABLE = "Receiver unavailable; nothing was retried. Try again later."
REDIRECTED = (
    "The receiver redirected; the redirect was not followed. Nothing was confirmed."
)
UNEXPECTED = "Unexpected answer (HTTP {status}); nothing was confirmed."
NO_RECEIPT = "No confirmed receipt; the submission may have arrived."

NO_RECEIVER = (
    "No receiver is configured in this build yet; the archive is written to a"
    " new local file instead"
)
# Without a receiver and without --write: the file's name beside the
# evaluation directory.
DEFAULT_FILE = "submission-<first 12 digits of the archive's SHA-256>.tar.gz"
NO_CONTACT = "No deletion contact is configured in this build"
NOTHING_SENT = "Nothing was sent."
NOTHING_WRITTEN = "Nothing was written."
