"""One request to the receiver and the message for its answer.

`http.client` reads no proxy settings and follows no redirects: the one POST
goes to the checked address or nowhere, and is never repeated. Connecting and
each read or write time out after 60 seconds; at most 16 KiB of the answer is
read, and only a 201 or 200 answer is read at all. No message holds the
address, a header, a body or an exception text.
"""

import http.client
import math
import re
import ssl
from dataclasses import dataclass

from victors_werkbank import __version__
from victors_werkbank.agent.transport import strict_json
from victors_werkbank.grading.evidence import sha256

from . import texts
from .receiver import check_address

TIMEOUT = 60.0
MAX_ANSWER_BYTES = 16_384
# The body is written in pieces, so the timeout bounds each write.
WRITE_BYTES = 65_536
_SECONDS = re.compile(r"[0-9]{1,12}")


@dataclass(frozen=True)
class Outcome:
    """What one send achieved: a confirmed receipt or not, and its message."""

    confirmed: bool
    message: str


def headers(body: bytes, installation_id: str) -> list[tuple[str, str]]:
    """The request's headers besides `Host`, in the order they are sent."""
    return [
        ("Content-Type", "application/gzip"),
        ("Content-Length", str(len(body))),
        ("X-Werkbank-Submission", texts.SUBMISSION_FORMAT),
        ("X-Werkbank-Installation", installation_id),
        ("X-Werkbank-Content-SHA256", sha256(body)),
        ("User-Agent", f"werkbank/{__version__}"),
        ("Accept", "application/json"),
    ]


def connection(address: str, timeout: float) -> tuple[http.client.HTTPConnection, str]:
    """An unopened connection to the checked receiver address and its path;
    HTTPS verifies the certificate and host name against the system's
    certificates."""
    parts = check_address(address)
    if parts.scheme == "https":
        return (
            http.client.HTTPSConnection(
                parts.host,
                parts.port,
                timeout=timeout,
                context=ssl.create_default_context(),
            ),
            parts.path,
        )
    return http.client.HTTPConnection(
        parts.host, parts.port, timeout=timeout
    ), parts.path


def _rate_limited(value: str | None) -> str:
    seconds = (value or "").strip(" \t")
    if _SECONDS.fullmatch(seconds) is None:
        return texts.RATE_LIMITED_LATER
    minutes = max(1, math.ceil(int(seconds) / 60))
    if minutes == 1:
        return texts.RATE_LIMITED_ONE
    return texts.RATE_LIMITED.format(minutes=minutes)


def _by_status(status: int, retry_after: str | None, raw_data: bool) -> str:
    if status == 400:
        return texts.MALFORMED
    if status == 413:
        if raw_data:
            return f"{texts.TOO_LARGE} {texts.WITHOUT_RAW}"
        return texts.TOO_LARGE
    if status == 415:
        return texts.UNSUPPORTED
    if status == 429:
        return _rate_limited(retry_after)
    if 500 <= status <= 599:
        return texts.UNAVAILABLE
    if 300 <= status <= 399:
        return texts.REDIRECTED
    return texts.UNEXPECTED.format(status=status)


def _receipt(status: int, data: bytes, digest: str) -> Outcome:
    """A 201 or 200 answer: confirmed only by a strict receipt of `digest`."""
    expected = "stored" if status == 201 else "duplicate"
    try:
        record = strict_json(data.decode("utf-8"))
    except (ValueError, UnicodeError):
        return Outcome(False, texts.NO_RECEIPT)
    if (
        not isinstance(record, dict)
        or set(record) != {"format", "status", "receipt"}
        or record["format"] != texts.RECEIPT_FORMAT
        or record["status"] != expected
        or record["receipt"] != digest
    ):
        return Outcome(False, texts.NO_RECEIPT)
    if status == 201:
        return Outcome(True, texts.RECEIVED.format(receipt=digest[:12]))
    return Outcome(True, texts.DUPLICATE)


def send(
    body: bytes,
    *,
    installation_id: str,
    address: str,
    raw_data: bool,
    timeout: float = TIMEOUT,
) -> Outcome:
    """POST `body` once to `address` and read the answer. AddressError, before
    any contact, for an address this client never contacts."""
    conn, path = connection(address, timeout)
    digest = sha256(body)
    try:
        conn.putrequest("POST", path, skip_accept_encoding=True)
        for name, value in headers(body, installation_id):
            conn.putheader(name, value)
        conn.endheaders()
        view = memoryview(body)
        try:
            for start in range(0, len(view), WRITE_BYTES):
                conn.send(view[start : start + WRITE_BYTES])
        except OSError:
            # A receiver may answer before it reads a body it refuses (too
            # large): an answer that arrived still decides the message.
            pass
        response = conn.getresponse()
        status = response.status
        if status in {200, 201}:
            data = response.read(MAX_ANSWER_BYTES + 1)
            if len(data) > MAX_ANSWER_BYTES:
                return Outcome(False, texts.NO_RECEIPT)
            return _receipt(status, data, digest)
        return Outcome(
            False, _by_status(status, response.getheader("Retry-After"), raw_data)
        )
    except (OSError, http.client.HTTPException, ValueError):
        return Outcome(False, texts.NO_RECEIPT)
    finally:
        conn.close()
