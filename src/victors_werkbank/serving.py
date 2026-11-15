"""Serving identity for reports; unknown parts are null with a reason.

The endpoint fingerprint is self-reported by the server, the owner note by the
owner and the local accelerators by the host's GPU tool. All are checked for
shape and consistency, never authenticated. A reported local filesystem path
is stored as its last component plus the SHA-256 of the full path, so no host
path is recorded.

Format v5 probes one engine endpoint for vLLM (`/version`), llama.cpp
(`/props`) and SGLang (`/get_server_info`) and keeps the engine settings that
response carries, the server's default chat template variables included; for
an endpoint on loopback it reads the product name and memory of this host's
NVIDIA accelerators; it records an optional hardware label, takes the rest of
the serving-host hardware from the owner note (format v2) and hashes the run
configuration (identity sampling, per-response cap, harness digest and
streaming mode) into the identity, so a streamed run never shares an identity
with a non-streamed one. Format v4 records, which have no default template
variables, and format v3 records, which have neither engine settings nor a
host section, are read unchanged, so their identities stay as recorded.
"""

import hashlib
import ipaddress
import json
import os
import re
import selectors
import shutil
import stat
import subprocess
import time
import urllib.parse
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeGuard

from .agent.measurements import (
    IDENTITY_SAMPLING_KEYS,
    validate_sampling,
    validate_template_kwargs,
)
from .agent.transport import EndpointError, strict_json

SERVING_FORMAT = "werkbank-serving-v5"
# Read unchanged: no default template variables.
SERVING_FORMAT_V4 = "werkbank-serving-v4"
# Read unchanged: no engine settings, no host section, and a digest only for a
# complete identity.
SERVING_FORMAT_V3 = "werkbank-serving-v3"
NOTE_FORMAT = "werkbank-serving-note-v2"
MODELS_BYTES = 1024 * 1024
VERSION_BYTES = 64 * 1024
# llama.cpp /props carries the chat template; SGLang reports all server args.
ENGINE_BYTES = 1024 * 1024
NOTE_BYTES = 16 * 1024
PROBE_SECONDS = 10.0
MAX_TEXT = 256
MAX_LISTED = 4096
MAX_CONTEXT_TOKENS = 16_777_216
SOFTWARE_KEYS = (
    "serving_runtime",
    "serving_version",
    "model_revision",
    "quantization",
    "kv_cache_dtype",
    "speculative_decoding",
    "context_profile",
    "chat_template",
    "tool_call_parser",
)
NOTE_ONLY_KEYS = (
    "model_revision",
    "quantization",
    "kv_cache_dtype",
    "speculative_decoding",
    "chat_template",
    "tool_call_parser",
)
# Serving-host hardware; all are part of a complete identity.
HARDWARE_KEYS = (
    "accelerator",
    "accelerator_memory_gib",
    "memory_type",
    "driver_version",
    "compute_runtime",
    "serving_host_os",
)
# Recorded and hashed when stated; release builds have no separate commit.
OPTIONAL_KEYS = ("engine_commit",)
NOTE_KEYS = SOFTWARE_KEYS + HARDWARE_KEYS + OPTIONAL_KEYS
MEMORY_TYPES = ("discrete", "unified")
MAX_MEMORY_GIB = 65536
# Hardware classes for evaluation across runs; extend here, never per run.
HARDWARE_CLASSES = {
    (24, "discrete"): "24gb-discrete",
    (32, "discrete"): "32gb-discrete",
    (96, "discrete"): "96gb-discrete",
    (128, "unified"): "128gb-unified",
}
# `owned_by` of /v1/models -> runtime name and its one read-only engine probe.
RUNTIMES = {
    "vllm": ("vllm", "version"),
    "llamacpp": ("llama.cpp", "props"),
    "sglang": ("sglang", "get_server_info"),
}
# Engine settings kept verbatim where the engine response carries them (the
# server arguments SGLang reports); null where it does not.
ENGINE_SETTINGS = ("quantization", "kv_cache_dtype", "speculative_decoding")
# The default chat template variables the server applies to every request that
# does not set them (SGLang's `default_chat_template_kwargs`), such as a
# model's thinking switch: part of the setup, as they change what the model is
# asked to do. They are kept under the rule for the variables a run sends
# (`validate_template_kwargs`), so no text, path or address is recorded; any
# other reported value is kept only as the SHA-256 of its canonical JSON, so
# servers with differing defaults never share an identity. `variables`, the
# digest or neither, with a fixed reason for the last two.
TEMPLATE_DEFAULTS = "chat_template_defaults"
TEMPLATE_DEFAULTS_KEYS = {"variables", "sha256", "reason"}
# The engine response carries none: vLLM's `/version` and llama.cpp's `/props`
# never do, and neither does a failed or unrequested probe.
DEFAULTS_NOT_REPORTED = "not reported"
# Reported as null or as an empty object: the server sets no default.
DEFAULTS_NONE = "none set"
DEFAULTS_OUTSIDE_RULE = "outside the template variable rule"
ENGINE_KEYS_V4 = {
    "runtime",
    "version",
    "context_tokens",
    "model_path",
    *ENGINE_SETTINGS,
    "reason",
}
ENGINE_KEYS = ENGINE_KEYS_V4 | {TEMPLATE_DEFAULTS}
ENGINE_KEYS_V3 = ENGINE_KEYS_V4 - set(ENGINE_SETTINGS)
HOST_KEYS = {"label", "accelerators", "reason"}
# This host's NVIDIA accelerators, read once for an endpoint on loopback: one
# bounded query of product name and total memory, never UUID, serial, bus,
# driver path or the tool's location.
GPU_TOOL = "nvidia-smi"
GPU_TOOL_ARGUMENTS = (
    "--query-gpu=name,memory.total",
    "--format=csv,noheader,nounits",
)
GPU_TOOL_SECONDS = 10.0
# Waiting for a stopped tool to exit is bounded too: a tool stuck in the
# driver may not die, and the run goes on without it.
GPU_TOOL_STOP_SECONDS = 1.0
GPU_TOOL_BYTES = 64 * 1024
MAX_ACCELERATORS = 64
# An owner-stated hardware label: short, ASCII, no path separator or newline.
MAX_LABEL = 64
_LABEL = re.compile(rf"[A-Za-z0-9][A-Za-z0-9 ._+()-]{{0,{MAX_LABEL - 1}}}")
_LOOPBACK_V4 = ipaddress.IPv4Network("127.0.0.0/8")
_LOOPBACK_V6 = ipaddress.IPv6Address("::1")
_MIB = re.compile(r"[0-9]+")
# The tool's bracketed text for a value it cannot give, such as `[N/A]` for
# the memory of a unified-memory GPU.
_PLACEHOLDER = re.compile(r"\[[A-Za-z][A-Za-z /]{0,62}\]")
# The run configuration hashed into the identity.
CONFIGURATION_KEYS = {"sampling", "max_tokens_per_call", "harness_sha256", "streaming"}
INVALID = "Invalid serving identity"
INVALID_NOTE = "Invalid serving note"
# A note's problems, from this fixed vocabulary only: a key is named with its
# rule, never a value, the file's path or its contents. An unknown key is
# named only when it is a plain identifier.
NOTE_UNREADABLE = "not a readable regular file (no symbolic link)"
NOTE_TOO_LARGE = f"larger than {NOTE_BYTES} bytes"
NOTE_NOT_JSON = "not strict JSON in UTF-8 (no duplicate keys, NaN or infinity)"
NOTE_NOT_OBJECT = "not a JSON object"
NOTE_UNKNOWN = "an unknown key"
_NOTE_KEY = re.compile(r"[A-Za-z0-9_]{1,64}")
_TEXT_RULE = f"must be printable text of 1 to {MAX_TEXT} characters, not a local path"
_NOTE_RULES = {
    "format": f"must be {NOTE_FORMAT}",
    "context_profile": f"must be an integer from 1 to {MAX_CONTEXT_TOKENS}",
    "accelerator_memory_gib": f"must be an integer from 1 to {MAX_MEMORY_GIB}",
    "memory_type": "must be " + " or ".join(MEMORY_TYPES),
    "model_revision": "must be 40 lowercase hex digits or sha256: and 64 lowercase"
    " hex digits",
}
INVALID_LABEL = "Invalid hardware label"

NOT_PROBED = "not probed"
NOT_LISTED = "model not listed"
INVALID_RESPONSE = "invalid response"
DEADLINE = "endpoint probe deadline exceeded"
REDIRECT = "endpoint redirect refused"
TOO_LARGE = "endpoint response too large"
TRANSPORT = "endpoint transport failed"
NOT_V1 = "endpoint base is not a /v1 path"
NOT_REQUESTED = "not requested"
OTHER_RUNTIME = "not requested for this runtime"
NOT_LOOPBACK = "endpoint not on loopback"
NO_GPU_TOOL = "no GPU tool found"
GPU_TOOL_FAILED = "GPU tool failed"
GPU_TOOL_DEADLINE = "GPU tool deadline exceeded"
INVALID_GPU_OUTPUT = "invalid GPU tool output"
NO_GPU = "no GPU reported"
_HOST_REASONS = frozenset(
    {
        NOT_PROBED,
        NOT_LOOPBACK,
        NO_GPU_TOOL,
        GPU_TOOL_FAILED,
        GPU_TOOL_DEADLINE,
        INVALID_GPU_OUTPUT,
        NO_GPU,
    }
)
_FAILURES = frozenset({DEADLINE, REDIRECT, TOO_LARGE, TRANSPORT, INVALID_RESPONSE})
_MESSAGES = {
    "Endpoint redirects are not supported": REDIRECT,
    "Response exceeds byte limit": TOO_LARGE,
    "Invalid JSON": INVALID_RESPONSE,
    "Endpoint base is not a /v1 path": NOT_V1,
}
_HTTP_REASON = re.compile(r"endpoint HTTP status [1-5][0-9]{2}")
_DRIVE = re.compile(r"[A-Za-z]:[\\/]")
_PATH_PREFIXES = ("/", "~", "./", "../", ".\\", "..\\", "\\")
_SEPARATORS = re.compile(r"[\\/]")
_REVISION = re.compile(r"[0-9a-f]{40}|sha256:[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ENTRY_KEYS = {"id", "owned_by", "root", "parent", "max_model_len"}
_IDENTITY_PARTS = {
    "models_entry",
    "serving_runtime",
    "serving_version",
    "context_profile",
    *NOTE_ONLY_KEYS,
    *HARDWARE_KEYS,
}
_CONFLICT_PARTS = {"serving_runtime", "serving_version", "context_profile"}


class _Invalid(Exception):
    """A response that does not have the documented shape."""


def _is_text(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and 0 < len(value) <= MAX_TEXT and value.isprintable()


def _is_count(value: object, low: int, high: int | None = None) -> bool:
    return type(value) is int and value >= low and (high is None or value <= high)


def is_local_path(value: str) -> bool:
    """True for absolute, home, relative, Windows, UNC and file: paths."""
    return (
        value.startswith(_PATH_PREFIXES)
        or value[:5].lower() == "file:"
        or _DRIVE.match(value) is not None
    )


def path_safe(value: str) -> str | dict[str, str]:
    """Keep IDs as reported; a local path becomes basename plus SHA-256."""
    if not is_local_path(value):
        return value
    parts = [part for part in _SEPARATORS.split(value) if part]
    return {
        "basename": parts[-1] if parts else "",
        "sha256": hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest(),
    }


def path_digest(value: str) -> dict[str, str]:
    """Basename plus SHA-256 of a value that is always a path, relative or not
    (an engine's model path), so no part of a host path is recorded."""
    parts = [part for part in _SEPARATORS.split(value) if part]
    return {
        "basename": parts[-1] if parts else "",
        "sha256": hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest(),
    }


def _note_value_valid(key: str, item: object) -> bool:
    if key == "context_profile":
        return _is_count(item, 1, MAX_CONTEXT_TOKENS)
    if key == "accelerator_memory_gib":
        return _is_count(item, 1, MAX_MEMORY_GIB)
    if key == "memory_type":
        return item in MEMORY_TYPES
    if not _is_text(item) or is_local_path(item):
        return False
    return key != "model_revision" or _REVISION.fullmatch(item) is not None


def note_problems(value: object) -> list[str]:
    """Every problem of an owner serving note, in key order: the format, the
    known keys in their order, then the unknown keys (sorted) and the allowed
    keys. Each names a key and its rule from a fixed vocabulary; no value is
    ever part of it."""
    if not isinstance(value, dict):
        return [NOTE_NOT_OBJECT]
    problems = []
    if value.get("format") != NOTE_FORMAT:
        problems.append(f"format {_NOTE_RULES['format']}")
    for key in NOTE_KEYS:
        if key in value and not _note_value_valid(key, value[key]):
            problems.append(f"{key} {_NOTE_RULES.get(key, _TEXT_RULE)}")
    unknown = [key for key in value if key not in {"format", *NOTE_KEYS}]
    named = sorted(
        key
        for key in unknown
        if isinstance(key, str) and _NOTE_KEY.fullmatch(key) is not None
    )
    problems += [f"unknown key {key}" for key in named]
    hidden = len(unknown) - len(named)
    if hidden:
        problems.append(NOTE_UNKNOWN if hidden == 1 else f"{hidden} unknown keys")
    if unknown:
        problems.append("allowed keys: " + ", ".join(("format", *NOTE_KEYS)))
    return problems


def validate_note(value: object) -> None:
    """Raise ValueError unless value is a well-formed owner serving note; a key
    it leaves out is missing from the identity. The message names every
    problem (`note_problems`), never a value."""
    problems = note_problems(value)
    if problems:
        raise ValueError(f"{INVALID_NOTE}: " + "; ".join(problems))


def hardware_class(note: dict[str, Any] | None) -> str | None:
    """Class of the noted serving-host memory, if it is a listed class."""
    if note is None:
        return None
    return HARDWARE_CLASSES.get(
        (note.get("accelerator_memory_gib"), note.get("memory_type"))  # type: ignore[arg-type]
    )


def is_hardware_label(value: object) -> TypeGuard[str]:
    """True for 1..64 ASCII letters, digits, spaces and `._+()-`, starting
    with a letter or digit and not ending in a space."""
    return (
        isinstance(value, str)
        and _LABEL.fullmatch(value) is not None
        and not value.endswith(" ")
    )


def validate_label(value: object) -> None:
    """Raise ValueError unless value is a well-formed hardware label."""
    if not is_hardware_label(value):
        raise ValueError(INVALID_LABEL)


def is_loopback(base_url: object) -> bool:
    """True when the endpoint URL's host is a loopback IP literal
    (127.0.0.0/8 or ::1) or the name `localhost`; no name is resolved."""
    if not isinstance(base_url, str):
        return False
    try:
        host = urllib.parse.urlsplit(base_url).hostname
    except ValueError:
        return False
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host or "")
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv4Address):
        return address in _LOOPBACK_V4
    return address == _LOOPBACK_V6


def _gib(memory: str) -> int | None:
    """Whole GiB from the tool's MiB, rounded; null for a placeholder such as
    `[N/A]` (unified-memory GPUs) and outside 1..MAX_MEMORY_GIB."""
    if _PLACEHOLDER.fullmatch(memory) is not None or len(memory) > 9:
        return None
    gib = (int(memory) + 512) // 1024
    return gib if 1 <= gib <= MAX_MEMORY_GIB else None


def _accelerator_order(item: dict[str, Any]) -> tuple[str, int]:
    memory = item["memory_gib"]
    return item["name"], -1 if memory is None else memory


def _is_product_name(value: object) -> TypeGuard[str]:
    """A GPU product name: plain text, not a path, not the tool's bracketed
    placeholder and without a colon, which a message carrying a bus ID has
    and no product name does."""
    return (
        _is_text(value)
        and not is_local_path(value)
        and not value.startswith("[")
        and ":" not in value
    )


def _accelerators(stdout: bytes) -> tuple[list[dict[str, Any]] | None, str | None]:
    """One `name, MiB` line per GPU, in name order; a fixed reason otherwise,
    also for a line whose memory is neither digits nor a placeholder."""
    try:
        lines = [line for line in stdout.decode("utf-8").splitlines() if line.strip()]
    except UnicodeDecodeError:
        return None, INVALID_GPU_OUTPUT
    if not lines:
        return None, NO_GPU
    if len(lines) > MAX_ACCELERATORS:
        return None, INVALID_GPU_OUTPUT
    found = []
    for line in lines:
        # A product name may hold a comma; the memory is the last field.
        name, comma, memory = line.rpartition(",")
        name, memory = name.strip(), memory.strip()
        if (
            not comma
            or not _is_product_name(name)
            or (_MIB.fullmatch(memory) or _PLACEHOLDER.fullmatch(memory)) is None
        ):
            return None, INVALID_GPU_OUTPUT
        found.append({"name": name, "memory_gib": _gib(memory)})
    return sorted(found, key=_accelerator_order), None


def _tool_output(process: Any) -> tuple[bytes | None, str | None]:
    """The tool's stdout once it exited 0, read in bounded chunks within the
    deadline, or a fixed reason; output beyond the cap is not read."""
    deadline = time.monotonic() + GPU_TOOL_SECONDS
    output = bytearray()
    descriptor = process.stdout.fileno()
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                return None, GPU_TOOL_DEADLINE
            chunk = os.read(descriptor, GPU_TOOL_BYTES + 1 - len(output))
            if not chunk:
                break
            output += chunk
            if len(output) > GPU_TOOL_BYTES:
                return None, INVALID_GPU_OUTPUT
    try:
        returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        return None, GPU_TOOL_DEADLINE
    if returncode != 0:
        return None, GPU_TOOL_FAILED
    return bytes(output), None


def _stop(process: Any) -> None:
    """Kill a tool still running and wait a bounded time for it; one that
    does not exit (stuck in the driver) is left behind, never waited for."""
    try:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=GPU_TOOL_STOP_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        pass
    finally:
        process.stdout.close()


def local_accelerators(
    *,
    popen: Callable[..., Any] = subprocess.Popen,
    which: Callable[[str], str | None] = shutil.which,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """This host's NVIDIA accelerators (product name, memory in GiB) from one
    bounded `nvidia-smi` query without a shell, or None with a fixed reason.
    The deadline and the output cap hold while the tool runs, and a tool that
    overruns them is killed. Never raises and is never retried; the tool's
    location and its error output are never recorded."""
    tool = which(GPU_TOOL)
    if not tool:
        return None, NO_GPU_TOOL
    try:
        process = popen(
            [tool, *GPU_TOOL_ARGUMENTS],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None, GPU_TOOL_FAILED
    try:
        stdout, reason = _tool_output(process)
    except (OSError, ValueError):
        stdout, reason = None, GPU_TOOL_FAILED
    finally:
        _stop(process)
    if stdout is None:
        return None, reason
    return _accelerators(stdout)


def _host(endpoint: Any, label: str | None) -> dict[str, Any]:
    """The serving host: the stated label and, for an endpoint on loopback,
    this host's accelerators. A tunnel or a remote forward on loopback is not
    detectable, and the tool lists every GPU, not only those the server uses."""
    base_url = getattr(endpoint, "base_url", None)
    accelerators: list[dict[str, Any]] | None = None
    if not isinstance(base_url, str):
        reason: str | None = NOT_PROBED
    elif not is_loopback(base_url):
        reason = NOT_LOOPBACK
    else:
        accelerators, reason = local_accelerators()
    return {"label": label, "accelerators": accelerators, "reason": reason}


def _read_note(descriptor: int) -> bytes | str:
    """The note's bytes, or the problem that keeps them from being read."""
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return NOTE_UNREADABLE
        chunks: list[bytes] = []
        total = 0
        while total <= NOTE_BYTES:
            chunk = os.read(descriptor, NOTE_BYTES + 1 - total)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
    except OSError:
        return NOTE_UNREADABLE
    return NOTE_TOO_LARGE if total > NOTE_BYTES else b"".join(chunks)


def load_note(path: Path) -> dict[str, Any]:
    """Read a bounded regular-file note. An error names the broken rule or the
    note's keys (`note_problems`), never the path or the contents, and chains
    no exception that could."""
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise ValueError(f"{INVALID_NOTE}: {NOTE_UNREADABLE}") from None
    try:
        data = _read_note(descriptor)
    finally:
        os.close(descriptor)
    if isinstance(data, str):
        raise ValueError(f"{INVALID_NOTE}: {data}") from None
    try:
        note = strict_json(data.decode("utf-8"))
    except (ValueError, RecursionError):
        raise ValueError(f"{INVALID_NOTE}: {NOTE_NOT_JSON}") from None
    problems = note_problems(note)
    if problems:
        raise ValueError(f"{INVALID_NOTE}: " + "; ".join(problems)) from None
    return note


def _failure(error: Exception, *, root: bool) -> str:
    """Fixed reason for a failed probe; the exception text is never recorded."""
    if isinstance(error, TimeoutError):
        return DEADLINE
    if isinstance(error, EndpointError):
        raw = error.raw
        if isinstance(raw, dict) and _is_count(raw.get("http_status"), 100, 599):
            return f"endpoint HTTP status {raw['http_status']}"
        reason = _MESSAGES.get(str(error), TRANSPORT)
        return TRANSPORT if reason == NOT_V1 and not root else reason
    return TRANSPORT


def _optional_text(item: dict[str, Any], key: str) -> str | None:
    value = item.get(key)
    if value is not None and not _is_text(value):
        raise _Invalid
    return value


def _project(item: dict[str, Any]) -> dict[str, Any]:
    """Whitelisted, path-safe entry fields; everything else is dropped."""
    identifier = item["id"]
    owned_by = _optional_text(item, "owned_by")
    root = _optional_text(item, "root")
    parent = _optional_text(item, "parent")
    length = item.get("max_model_len")
    if (
        not _is_text(identifier)
        or (owned_by is not None and is_local_path(owned_by))
        or (length is not None and not _is_count(length, 1))
    ):
        raise _Invalid
    return {
        "id": path_safe(identifier),
        "owned_by": owned_by,
        "root": None if root is None else path_safe(root),
        "parent": None if parent is None else path_safe(parent),
        "max_model_len": length,
    }


def _select(response: object, model: str) -> tuple[dict[str, Any] | None, int]:
    if not isinstance(response, dict):
        raise _Invalid
    data = response.get("data")
    if not isinstance(data, list) or len(data) > MAX_LISTED:
        raise _Invalid
    if not all(isinstance(i, dict) and isinstance(i.get("id"), str) for i in data):
        raise _Invalid
    matches = [item for item in data if item["id"] == model]
    if len(matches) > 1:
        raise _Invalid
    return (_project(matches[0]) if matches else None), len(data)


def _probe_models(probe: Any, model: str) -> dict[str, Any]:
    try:
        response = probe(
            "models", root=False, max_bytes=MODELS_BYTES, timeout=PROBE_SECONDS
        )
    except (OSError, ValueError) as error:
        return {"entry": None, "listed": None, "reason": _failure(error, root=False)}
    try:
        entry, listed = _select(response, model)
    except _Invalid:
        return {"entry": None, "listed": None, "reason": INVALID_RESPONSE}
    reason = None if entry is not None else NOT_LISTED
    return {"entry": entry, "listed": listed, "reason": reason}


def _no_defaults(reason: str) -> dict[str, Any]:
    return {"variables": None, "sha256": None, "reason": reason}


def _engine_failure(
    runtime: str | None, reason: str, keys: set[str] = ENGINE_KEYS
) -> dict[str, Any]:
    failure = {
        "runtime": runtime,
        "version": None,
        "context_tokens": None,
        "model_path": None,
        **dict.fromkeys(ENGINE_SETTINGS),
        TEMPLATE_DEFAULTS: _no_defaults(DEFAULTS_NOT_REPORTED),
        "reason": reason,
    }
    return {key: value for key, value in failure.items() if key in keys}


def _field(response: dict[str, Any], key: str, kind: str) -> Any:
    """An optional engine field: absent is None, a malformed one is invalid."""
    value = response.get(key)
    if value is None:
        return None
    if kind == "count":
        if not _is_count(value, 1, MAX_CONTEXT_TOKENS):
            raise _Invalid
        return value
    if not _is_text(value) or (kind == "plain" and is_local_path(value)):
        raise _Invalid
    return value if kind == "plain" else path_digest(value)


def _template_defaults(response: dict[str, Any]) -> dict[str, Any]:
    """SGLang's default template variables: kept within the rule, none set
    for null or an empty object, otherwise only the digest of the value. A
    value too deep to digest is malformed."""
    if "default_chat_template_kwargs" not in response:
        return _no_defaults(DEFAULTS_NOT_REPORTED)
    value = response["default_chat_template_kwargs"]
    if value is None or (type(value) is dict and not value):
        return _no_defaults(DEFAULTS_NONE)
    try:
        return {
            "variables": validate_template_kwargs(value),
            "sha256": None,
            "reason": None,
        }
    except ValueError:
        pass
    try:
        found = digest(value)
    except (ValueError, TypeError, RecursionError):
        raise _Invalid from None
    return {"variables": None, "sha256": found, "reason": DEFAULTS_OUTSIDE_RULE}


def _extract(path: str, response: object) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise _Invalid
    unreported = {
        **dict.fromkeys(ENGINE_SETTINGS),
        TEMPLATE_DEFAULTS: _no_defaults(DEFAULTS_NOT_REPORTED),
    }
    if path == "version":
        version = _field(response, "version", "plain")
        if version is None:
            raise _Invalid
        return {
            "version": version,
            "context_tokens": None,
            "model_path": None,
            **unreported,
        }
    if path == "props":
        settings = response.get("default_generation_settings")
        if settings is not None and not isinstance(settings, dict):
            raise _Invalid
        return {
            "version": _field(response, "build_info", "plain"),
            # Per-slot context: what one request can use.
            "context_tokens": _field(settings or {}, "n_ctx", "count"),
            "model_path": _field(response, "model_path", "name"),
            **unreported,
        }
    # SGLang's server arguments, kept verbatim (`auto` included); an absent or
    # null argument is unknown.
    return {
        "version": _field(response, "version", "plain"),
        "context_tokens": _field(response, "context_length", "count"),
        "model_path": _field(response, "model_path", "name"),
        "quantization": _field(response, "quantization", "plain"),
        "kv_cache_dtype": _field(response, "kv_cache_dtype", "plain"),
        "speculative_decoding": _field(response, "speculative_algorithm", "plain"),
        TEMPLATE_DEFAULTS: _template_defaults(response),
    }


def _probe_engine(probe: Any, entry: dict[str, Any] | None) -> dict[str, Any]:
    """At most one read-only engine probe, chosen by the reported `owned_by`."""
    if entry is None:
        return _engine_failure(None, NOT_REQUESTED)
    known = RUNTIMES.get(entry["owned_by"])  # type: ignore[arg-type]
    if known is None:
        return _engine_failure(None, OTHER_RUNTIME)
    runtime, path = known
    limit = VERSION_BYTES if path == "version" else ENGINE_BYTES
    try:
        response = probe(path, root=True, max_bytes=limit, timeout=PROBE_SECONDS)
    except (OSError, ValueError) as error:
        return _engine_failure(runtime, _failure(error, root=True))
    try:
        fields = _extract(path, response)
    except _Invalid:
        return _engine_failure(runtime, INVALID_RESPONSE)
    return {"runtime": runtime, **fields, "reason": None}


def collect_serving(
    endpoint: Any,
    model: str,
    note: dict[str, Any] | None,
    *,
    configuration: dict[str, Any],
    hardware_label: str | None = None,
) -> dict[str, Any]:
    """Probe GET /v1/models and at most one engine endpoint once each and,
    for an endpoint on loopback, query this host's GPU tool once; never
    retry. `configuration` is the run's identity sampling, per-response cap,
    harness digest and streaming mode."""
    if note is not None:
        validate_note(note)
    if hardware_label is not None:
        validate_label(hardware_label)
    _check_configuration(configuration)
    probe = getattr(endpoint, "get_json", None)
    models: dict[str, Any]
    engine: dict[str, Any]
    if not callable(probe):
        models = {"entry": None, "listed": None, "reason": NOT_PROBED}
        engine = _engine_failure(None, NOT_PROBED)
    else:
        models = _probe_models(probe, model)
        engine = _probe_engine(probe, models["entry"])
    host = _host(endpoint, hardware_label)
    return {
        "format": SERVING_FORMAT,
        "models": models,
        "engine": engine,
        "host": host,
        "owner_note": note,
        "hardware_class": hardware_class(note),
        "configuration": configuration,
        "identity": identity(
            models, engine, note, configuration, host=host, model_id=model
        ),
    }


def _gaps(
    models: dict[str, Any], engine: dict[str, Any], note: dict[str, Any] | None
) -> tuple[list[str], list[str]]:
    """What a complete identity still misses, and where note and server
    disagree; detected engine settings and host values never fill a gap."""
    entry = models["entry"]
    settings = note or {}
    runtime, version = engine["runtime"], engine["version"]
    missing = [key for key in (*NOTE_ONLY_KEYS, *HARDWARE_KEYS) if key not in settings]
    if entry is None:
        missing.append("models_entry")
    if runtime is None and "serving_runtime" not in settings:
        missing.append("serving_runtime")
    if version is None and "serving_version" not in settings:
        missing.append("serving_version")
    lengths = {
        value
        for value in (
            entry["max_model_len"] if entry is not None else None,
            engine["context_tokens"],
            settings.get("context_profile"),
        )
        if value is not None
    }
    if not lengths:
        missing.append("context_profile")
    conflicts = []
    if runtime is not None and settings.get("serving_runtime", runtime) != runtime:
        conflicts.append("serving_runtime")
    if version is not None and settings.get("serving_version", version) != version:
        conflicts.append("serving_version")
    if len(lengths) > 1:
        conflicts.append("context_profile")
    return sorted(missing), sorted(conflicts)


def digest(parts: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON text of `parts` (sorted keys, ASCII,
    no spaces), as every identity digest is taken."""
    canonical = json.dumps(
        parts,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _engine_parts(engine: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    """The engine fields and settings an identity digest covers: every key
    of the format but the reasons, the probe's and the template defaults'."""
    parts = {k: engine[k] for k in sorted(keys - {"reason"})}
    if TEMPLATE_DEFAULTS in parts:
        defaults = parts[TEMPLATE_DEFAULTS]
        parts[TEMPLATE_DEFAULTS] = {
            "variables": defaults["variables"],
            "sha256": defaults["sha256"],
        }
    return parts


def identity(
    models: dict[str, Any],
    engine: dict[str, Any],
    note: dict[str, Any] | None,
    configuration: dict[str, Any],
    *,
    host: dict[str, Any],
    model_id: str,
    engine_keys: set[str] = ENGINE_KEYS,
) -> dict[str, Any]:
    """Completeness, consistency and digest of a serving identity
    (`werkbank-serving-v5`, or `werkbank-serving-v4` with that format's
    `engine_keys`). The digest is always present: it covers the model entry
    (the served model name where no entry was read), the engine fields and
    settings, the server's default template variables included, the host's
    label and accelerators, the note and the configuration, never a reason."""
    entry = models["entry"]
    missing, conflicts = _gaps(models, engine, note)
    parts = {
        "models_entry": entry,
        "engine": _engine_parts(engine, engine_keys),
        "host": {"label": host["label"], "accelerators": host["accelerators"]},
        "owner_note": note,
        "configuration": configuration,
    }
    if entry is None:
        parts["model_id"] = model_id
    return {
        "complete": not missing and not conflicts,
        "missing": missing,
        "conflicts": conflicts,
        "sha256": digest(parts),
    }


def _parts_v3(
    models: dict[str, Any],
    engine: dict[str, Any],
    note: dict[str, Any] | None,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """What a `werkbank-serving-v3` identity digest covers."""
    return {
        "models_entry": models["entry"],
        "engine": {k: engine[k] for k in sorted(ENGINE_KEYS_V3 - {"reason"})},
        "owner_note": note,
        "configuration": configuration,
    }


def identity_v3(
    models: dict[str, Any],
    engine: dict[str, Any],
    note: dict[str, Any] | None,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """Completeness, consistency and digest of a `werkbank-serving-v3`
    identity, as that format recorded it: a digest only when complete."""
    missing, conflicts = _gaps(models, engine, note)
    complete = not missing and not conflicts
    recorded = None
    if complete:
        recorded = digest(_parts_v3(models, engine, note, configuration))
    return {
        "complete": complete,
        "missing": missing,
        "conflicts": conflicts,
        "sha256": recorded,
    }


def fingerprint(value: dict[str, Any], model_id: str) -> str:
    """The setup fingerprint of a validated serving record of the served
    model `model_id`: the identity digest, which formats v5 and v4 always
    record and format v3 records for a complete identity. For an incomplete v3
    identity it is the digest of exactly what a complete one covers, with
    the served model name where no models entry was read, so runs of one
    recorded setup share it whether or not their identity is complete."""
    recorded = value["identity"]["sha256"]
    if recorded is not None:
        return recorded
    parts = _parts_v3(
        value["models"], value["engine"], value["owner_note"], value["configuration"]
    )
    if parts["models_entry"] is None:
        parts["model_id"] = model_id
    return digest(parts)


def _check(condition: bool) -> None:
    if not condition:
        raise ValueError(INVALID)


def _section(value: object, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(INVALID)
    return value


def _check_plain(value: object) -> None:
    _check(_is_text(value) and not is_local_path(value))


def _check_name(value: object) -> None:
    """A reported ID or path: plain text, or exactly basename plus SHA-256."""
    if isinstance(value, dict):
        fields = _section(value, {"basename", "sha256"})
        basename = fields["basename"]
        _check(
            isinstance(basename, str)
            and len(basename) <= MAX_TEXT
            and basename.isprintable()
            and _SEPARATORS.search(basename) is None
        )
        _check(isinstance(fields["sha256"], str))
        _check(_SHA256.fullmatch(fields["sha256"]) is not None)
    else:
        _check_plain(value)


def _check_entry(entry: object) -> None:
    fields = _section(entry, _ENTRY_KEYS)
    _check_name(fields["id"])
    if fields["owned_by"] is not None:
        _check_plain(fields["owned_by"])
    for key in ("root", "parent"):
        if fields[key] is not None:
            _check_name(fields[key])
    _check(fields["max_model_len"] is None or _is_count(fields["max_model_len"], 1))


def _is_failure(reason: object) -> bool:
    return isinstance(reason, str) and (
        reason in _FAILURES or _HTTP_REASON.fullmatch(reason) is not None
    )


def _check_models(models: object) -> dict[str, Any]:
    fields = _section(models, {"entry", "listed", "reason"})
    entry, listed, reason = fields["entry"], fields["listed"], fields["reason"]
    if entry is not None:
        _check_entry(entry)
        _check(reason is None and _is_count(listed, 1, MAX_LISTED))
    elif reason == NOT_LISTED:
        _check(_is_count(listed, 0, MAX_LISTED))
    else:
        _check(listed is None and (reason == NOT_PROBED or _is_failure(reason)))
    return fields


def _check_configuration(value: object) -> None:
    fields = _section(value, CONFIGURATION_KEYS)
    sampling = fields["sampling"]
    try:
        validate_sampling(sampling)
    except ValueError:
        raise ValueError(INVALID) from None
    _check(set(sampling) <= set(IDENTITY_SAMPLING_KEYS))
    _check(_is_count(fields["max_tokens_per_call"], 1, 65536))
    digest = fields["harness_sha256"]
    _check(isinstance(digest, str) and _SHA256.fullmatch(digest) is not None)
    # A boolean, never a number that compares equal to one.
    _check(type(fields["streaming"]) is bool)


def _check_engine(
    engine: object, models: dict[str, Any], keys: set[str]
) -> dict[str, Any]:
    fields = _section(engine, keys)
    runtime, reason = fields["runtime"], fields["reason"]
    entry = models["entry"]
    known = RUNTIMES.get(entry["owned_by"]) if entry is not None else None
    if models["reason"] == NOT_PROBED:
        _check(fields == _engine_failure(None, NOT_PROBED, keys))
    elif entry is None:
        _check(fields == _engine_failure(None, NOT_REQUESTED, keys))
    elif known is None:
        _check(fields == _engine_failure(None, OTHER_RUNTIME, keys))
    elif reason is not None:
        _check(runtime == known[0] and (reason == NOT_V1 or _is_failure(reason)))
        _check(fields == _engine_failure(runtime, reason, keys))
    else:
        _check(runtime == known[0])
        if fields["version"] is not None:
            _check_plain(fields["version"])
        _check(known[1] != "version" or fields["version"] is not None)
        context, path = fields["context_tokens"], fields["model_path"]
        _check(context is None or _is_count(context, 1, MAX_CONTEXT_TOKENS))
        if path is not None:
            _check(isinstance(path, dict))
            _check_name(path)
        if known[1] == "version":
            _check(context is None and path is None)
        # Only SGLang's server arguments carry engine settings.
        for key in sorted(keys & set(ENGINE_SETTINGS)):
            if fields[key] is not None:
                _check(known[1] == "get_server_info")
                _check_plain(fields[key])
        if TEMPLATE_DEFAULTS in keys:
            _check_defaults(fields[TEMPLATE_DEFAULTS], known[1])
    return fields


def _check_defaults(value: object, path: str) -> None:
    """The server's default template variables: within the rule, a digest
    with its reason, or neither with its reason; only SGLang reports any."""
    fields = _section(value, TEMPLATE_DEFAULTS_KEYS)
    variables, found, reason = fields["variables"], fields["sha256"], fields["reason"]
    if path != "get_server_info":
        _check(fields == _no_defaults(DEFAULTS_NOT_REPORTED))
    elif variables is not None:
        try:
            validate_template_kwargs(variables)
        except ValueError:
            raise ValueError(INVALID) from None
        _check(found is None and reason is None)
    elif found is not None:
        _check(isinstance(found, str) and _SHA256.fullmatch(found) is not None)
        _check(reason == DEFAULTS_OUTSIDE_RULE)
    else:
        _check(reason in (DEFAULTS_NOT_REPORTED, DEFAULTS_NONE))


def _check_host(host: object) -> dict[str, Any]:
    fields = _section(host, HOST_KEYS)
    label, accelerators, reason = (
        fields["label"],
        fields["accelerators"],
        fields["reason"],
    )
    _check(label is None or is_hardware_label(label))
    if reason is not None:
        _check(isinstance(reason, str) and reason in _HOST_REASONS)
        _check(accelerators is None)
        return fields
    _check(isinstance(accelerators, list))
    _check(1 <= len(accelerators) <= MAX_ACCELERATORS)
    for item in accelerators:
        part = _section(item, {"name", "memory_gib"})
        _check(_is_product_name(part["name"]))
        memory = part["memory_gib"]
        _check(memory is None or _is_count(memory, 1, MAX_MEMORY_GIB))
    _check(accelerators == sorted(accelerators, key=_accelerator_order))
    return fields


def _check_identity_shape(value: object) -> dict[str, Any]:
    recorded = _section(value, {"complete", "missing", "conflicts", "sha256"})
    _check(type(recorded["complete"]) is bool)
    for key, names in (("missing", _IDENTITY_PARTS), ("conflicts", _CONFLICT_PARTS)):
        items = recorded[key]
        _check(
            isinstance(items, list)
            and all(isinstance(i, str) and i in names for i in items)
        )
    _check(recorded["sha256"] is None or isinstance(recorded["sha256"], str))
    return recorded


def _check_note(note: object) -> None:
    if note is not None:
        try:
            validate_note(note)
        except ValueError:
            raise ValueError(INVALID) from None


_V3_KEYS = {
    "format",
    "models",
    "engine",
    "owner_note",
    "hardware_class",
    "configuration",
    "identity",
}


def _validate_v3(value: dict[str, Any]) -> None:
    """A `werkbank-serving-v3` record exactly as that format wrote it."""
    fields = _section(value, _V3_KEYS)
    _check(fields["format"] == SERVING_FORMAT_V3)
    models = _check_models(fields["models"])
    engine = _check_engine(fields["engine"], models, ENGINE_KEYS_V3)
    note = fields["owner_note"]
    _check_note(note)
    _check(fields["hardware_class"] == hardware_class(note))
    _check_configuration(fields["configuration"])
    recorded = _check_identity_shape(fields["identity"])
    _check(recorded == identity_v3(models, engine, note, fields["configuration"]))


def validate_serving(value: object, model_id: object) -> None:
    """Raise ValueError unless value has exactly the collect_serving shape
    for the run's served model `model_id`, or is a `werkbank-serving-v4` or
    `werkbank-serving-v3` record as that format wrote it."""
    if isinstance(value, dict) and value.get("format") == SERVING_FORMAT_V3:
        _validate_v3(value)
        return
    fields = _section(value, _V3_KEYS | {"host"})
    formats = {SERVING_FORMAT: ENGINE_KEYS, SERVING_FORMAT_V4: ENGINE_KEYS_V4}
    _check(isinstance(fields["format"], str) and fields["format"] in formats)
    keys = formats[fields["format"]]
    if not isinstance(model_id, str):
        raise ValueError(INVALID)
    models = _check_models(fields["models"])
    engine = _check_engine(fields["engine"], models, keys)
    host = _check_host(fields["host"])
    note = fields["owner_note"]
    _check_note(note)
    _check(fields["hardware_class"] == hardware_class(note))
    _check_configuration(fields["configuration"])
    recorded = _check_identity_shape(fields["identity"])
    expected = identity(
        models,
        engine,
        note,
        fields["configuration"],
        host=host,
        model_id=model_id,
        engine_keys=keys,
    )
    _check(recorded == expected)


def context_tokens(value: dict[str, Any]) -> int | None:
    """Smallest context the server, its engine or the owner note reports
    (validated record). The smallest value fails closed when they disagree.
    """
    entry = value["models"]["entry"]
    note = value["owner_note"]
    reported = [
        entry["max_model_len"] if entry is not None else None,
        note.get("context_profile") if note is not None else None,
    ]
    reported.append(value["engine"]["context_tokens"])
    known = [tokens for tokens in reported if tokens is not None]
    return min(known) if known else None
