"""Finding an OpenAI-compatible server on this computer, read-only.

Each usual port gets exactly one `GET http://127.0.0.1:<port>/v1/models`, in
the fixed order below, through the run's own endpoint transport: proxy
settings ignored, redirects refused and never followed, no credentials, a
bounded strict-JSON read in a disposable worker under a total deadline of
`PROBE_SECONDS`, no retry. No other host, path or request; nothing found is
recorded, and model names are only shown, never one that names a file. Any
answer other than a model list counts as no server; HTTP 401 or 403 is shown
as a server that needs a key. The search never sends an API key, also where
`WERKBANK_API_KEY` is set: a key goes only to an endpoint the tester named,
whose model list `models` then reads with it.
"""

from dataclasses import dataclass
from typing import Any

from victors_werkbank.agent.model_process import IsolatedEndpoint
from victors_werkbank.agent.transport import EndpointError
from victors_werkbank.serving import MAX_LISTED, MODELS_BYTES, is_local_path

# (port, the servers that usually listen there), in probing order. Gateways
# and web interfaces are left out: they need not pass usage through.
PORTS: tuple[tuple[int, str], ...] = (
    (8000, "vLLM"),
    (8080, "llama.cpp"),
    (30000, "SGLang"),
    (11434, "Ollama"),
    (1234, "LM Studio"),
    (5000, "TabbyAPI or text-generation-webui"),
    (5001, "KoboldCpp"),
    (1337, "Jan"),
)
USUAL_PORTS = tuple(port for port, _ in PORTS)
HOST = "127.0.0.1"
PROBE_SECONDS = 3.0
# The probe worker carries a model name; a GET never sends it.
PLACEHOLDER = "werkbank-discovery"
NEEDS_KEY = (401, 403)
# The endings of a model file: a listed ID with one (in any case) names a file
# rather than a served model, as does one with a backslash, so discovery
# counts it with the paths. `werkbank run` admits `--model` by its own rule.
MODEL_FILE_SUFFIXES = (".gguf", ".safetensors", ".bin", ".pt", ".pth")


@dataclass(frozen=True)
class Listing:
    """What one base URL's model list offers: the names a run can use, in
    the listed order, and how many listed IDs name a file (`names_a_file`),
    which are never offered, chosen or shown."""

    base_url: str
    models: tuple[str, ...]
    paths: int


@dataclass(frozen=True)
class Found:
    """The servers that answered with a model list, and the base URLs of
    those that asked for a key."""

    servers: tuple[Listing, ...]
    locked: tuple[str, ...]


def base_url(port: int) -> str:
    return f"http://{HOST}:{port}/v1"


def usual_server(port: int) -> str | None:
    return dict(PORTS).get(port)


class NoList(Exception):
    """No model list: `locked` where the server answered 401 or 403."""

    def __init__(self, locked: bool):
        super().__init__("no model list")
        self.locked = locked


def names_a_file(name: str) -> bool:
    """Whether a listed model ID names a file: a local path (as a run
    refuses it), an ID with a backslash or one that ends in a model file's
    suffix (`MODEL_FILE_SUFFIXES`)."""
    return (
        is_local_path(name)
        or "\\" in name
        or name.lower().endswith(MODEL_FILE_SUFFIXES)
    )


def _names(response: Any) -> tuple[tuple[str, ...], int]:
    """The usable model names of a `/models` answer and the number of
    path-like IDs; NoList for anything but a model list."""
    if not isinstance(response, dict):
        raise NoList(False)
    data = response.get("data")
    if not isinstance(data, list) or len(data) > MAX_LISTED:
        raise NoList(False)
    if not all(isinstance(i, dict) and isinstance(i.get("id"), str) for i in data):
        raise NoList(False)
    names: list[str] = []
    paths = 0
    for item in data:
        name = item["id"]
        if not name or not name.isprintable():
            continue
        if names_a_file(name):
            paths += 1
        elif name not in names:
            names.append(name)
    return tuple(names), paths


def models(
    url: str, *, timeout: float = PROBE_SECONDS, api_key: str | None = None
) -> Listing:
    """One read-only `GET {url}/models`; NoList when no model list arrives.
    ValueError for a URL the transport refuses, before any contact. With
    `api_key`, for an endpoint the tester named, the request carries it."""
    endpoint = IsolatedEndpoint(url, PLACEHOLDER, api_key=api_key)
    try:
        response = endpoint.get_json(
            "models", root=False, max_bytes=MODELS_BYTES, timeout=timeout
        )
    except EndpointError as error:
        raw = error.raw
        status = raw.get("http_status") if isinstance(raw, dict) else None
        raise NoList(status in NEEDS_KEY) from None
    except OSError:
        raise NoList(False) from None
    names, paths = _names(response)
    return Listing(url, names, paths)


def discover(ports: tuple[int, ...], *, timeout: float = PROBE_SECONDS) -> Found:
    """Probe `ports` on 127.0.0.1 in the given order, each once and without
    an API key: whatever listens there was not named by the tester."""
    servers: list[Listing] = []
    locked: list[str] = []
    for port in ports:
        url = base_url(port)
        try:
            servers.append(models(url, timeout=timeout))
        except NoList as error:
            if error.locked:
                locked.append(url)
    return Found(tuple(servers), tuple(locked))
