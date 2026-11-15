"""The active profile: categories, weights, planned counts and, per card, its
title, difficulty label and summary; not executor admission. The cards' design
texts are no part of the package."""

import hashlib
import json
from importlib.resources import files


def _profile_bytes() -> bytes:
    return files("victors_werkbank").joinpath("profiles/core-v0.3.json").read_bytes()


def design_profile() -> dict:
    return json.loads(_profile_bytes())


def design_profile_sha256() -> str:
    return hashlib.sha256(_profile_bytes()).hexdigest()
