"""The receiver this build sends submissions to, and its address rules.

`ADDRESS` is the one address `werkbank submit` contacts and `DELETION_CONTACT`
the contact the summary names for deletion requests; neither is secret. While
`ADDRESS` is None, this build has no receiver: a submission can only be
written to a local file. Both live outside the harness digest, so setting them
never changes the Werkbank version.
"""

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

ADDRESS: str | None = "https://submit.victorswerkbank.workers.dev/submissions"
DELETION_CONTACT: str | None = "werkbank-loeschung.jeep654@passinbox.com"

# Plain HTTP only to this exact host: the tests' loopback receiver.
LOOPBACK = "127.0.0.1"
_PRINTABLE = re.compile(r"[!-~]+")


class AddressError(ValueError):
    """A receiver address this client never contacts."""


@dataclass(frozen=True)
class Address:
    scheme: str
    host: str
    port: int
    path: str


def check_address(address: str) -> Address:
    """The parts of a receiver address the client may contact: HTTPS to any
    host, plain HTTP only to exactly 127.0.0.1; no user information, query or
    fragment. AddressError otherwise, before any contact."""
    if not isinstance(address, str) or _PRINTABLE.fullmatch(address) is None:
        raise AddressError("The receiver address is not a plain URL")
    parts = urlsplit(address)
    if parts.scheme not in {"https", "http"}:
        raise AddressError("The receiver address is neither HTTPS nor loopback HTTP")
    if "@" in parts.netloc or "?" in address or "#" in address:
        raise AddressError(
            "The receiver address holds user information, a query or a fragment"
        )
    try:
        host, port = parts.hostname, parts.port
    except ValueError:
        raise AddressError("The receiver address has an invalid port") from None
    if not host:
        raise AddressError("The receiver address has no host")
    if parts.scheme == "http" and parts.netloc.split(":")[0] != LOOPBACK:
        raise AddressError("Plain HTTP is allowed only to 127.0.0.1")
    if port == 0:
        raise AddressError("The receiver address has an invalid port")
    default = 443 if parts.scheme == "https" else 80
    return Address(parts.scheme, host, port or default, parts.path or "/")
