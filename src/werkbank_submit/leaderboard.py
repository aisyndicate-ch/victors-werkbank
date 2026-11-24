"""The form of a display name for the leaderboard: what the client accepts
before it asks the privacy check, and what the receiver's side can check
again with the same function.

A name is the typed line without spaces at either end, in Unicode
normalization form NFC: 1 to 32 characters, each a letter (any Unicode
letter, umlauts included), a digit 0-9, a space or one of `-`, `_` and `.`,
the first and the last a letter or digit. So a name holds no `@`, `:` or
`/`, never an e-mail address or a URL, and no control or format character.
An empty line is no name: the entry shows "anonymous".
"""

import unicodedata

MAX_NAME_CHARS = 32
_DIGITS = frozenset("0123456789")
_JOINERS = frozenset(" -_.")


class NameRefused(ValueError):
    """A display name outside the accepted form."""


def _letter_or_digit(char: str) -> bool:
    return char in _DIGITS or unicodedata.category(char).startswith("L")


def display_name(text: str) -> str | None:
    """The display name of a typed `text`: None for an empty answer, the
    normalized name for an accepted one; NameRefused otherwise."""
    name = unicodedata.normalize("NFC", text.strip(" "))
    if name == "":
        return None
    if (
        len(name) > MAX_NAME_CHARS
        or not _letter_or_digit(name[0])
        or not _letter_or_digit(name[-1])
        or not all(_letter_or_digit(char) or char in _JOINERS for char in name)
    ):
        raise NameRefused(name)
    return name
