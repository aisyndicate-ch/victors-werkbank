"""Harbor pin and preflight, the pinned image reference, and the offline M02
smoke control.

The smoke control runs no model, gives no benchmark score and accepts no user
endpoint.
"""

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

HARBOR_VERSION = "0.24.0"
HARBOR_REVISION = "36232c205abfa42a5df4eee664e36dab0908acb6"
HARBOR_ARCHIVE_SHA256 = (
    "3af5bb021971568d84177dfd00aa820c7c00193339b3ca4cd166ad6df83fd814"
)
HARBOR_URL = (
    "https://github.com/harbor-framework/harbor/archive/" + HARBOR_REVISION + ".tar.gz"
)
SMOKE_IMAGE = "ghcr.io/aisyndicate-ch/werkbank-runtime@sha256:2c57f1598aa42b357d8220880591ca5c09939621ed5c12d12c05a1f6c0521f36"
MAX_EVIDENCE_BYTES = 8 * 1024 * 1024
# Package modules the smoke control's M02 grader imports inside the task
# container; every module `harbor_controls` reaches must be listed.
GRADER_MODULES = (
    "__init__.py",
    "agent/__init__.py",
    "agent/measurements.py",
    "agent/transport.py",
    "domains/__init__.py",
    "domains/algebra.py",
    "domains/variants.py",
    "harbor_controls.py",
)


class HarborError(RuntimeError):
    """Missing prerequisite or an unexpected integration outcome."""


# A pinned image: a local sha256 ID or repository@sha256 digest.
IMAGE = re.compile(r"(?:[a-zA-Z0-9][a-zA-Z0-9._/:-]*@)?sha256:[a-f0-9]{64}")


def image_reference(value: str) -> str:
    """A run's image is pinned: a mutable tag must never trigger an implicit
    pull."""
    if not IMAGE.fullmatch(value):
        raise HarborError("Use a local sha256 image ID or a repository@sha256 digest")
    return value


def preflight(image: str = SMOKE_IMAGE) -> dict[str, Any]:
    """Read-only; no installation, image build, model contact or credentials."""
    image_reference(image)
    errors = []
    if sys.version_info < (3, 12):
        errors.append("Harbor requires Python >=3.12")
    version = None
    try:
        distribution = importlib.metadata.distribution("harbor")
        version = distribution.version
        origin = json.loads(distribution.read_text("direct_url.json") or "{}")
        archive = origin.get("archive_info") if isinstance(origin, dict) else None
        hashes = archive.get("hashes") if isinstance(archive, dict) else None
        digest = hashes.get("sha256") if isinstance(hashes, dict) else None
        if digest is None and isinstance(archive, dict):
            legacy_hash = archive.get("hash")
            if isinstance(legacy_hash, str) and legacy_hash.startswith("sha256="):
                digest = legacy_hash.removeprefix("sha256=")
        if (
            version != HARBOR_VERSION
            or not isinstance(origin, dict)
            or origin.get("url") != HARBOR_URL
            or digest != HARBOR_ARCHIVE_SHA256
        ):
            errors.append("Install the exact archive in requirements-harbor.txt")
    except (importlib.metadata.PackageNotFoundError, ValueError):
        errors.append("Install the hash-locked requirements-harbor.txt")
    docker = shutil.which("docker")
    if docker is None:
        errors.append("Docker CLI is unavailable")
    else:
        try:
            result = subprocess.run(
                [docker, "image", "inspect", image, "--format", "{{.Id}}"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if result.returncode:
                errors.append(
                    "Docker access or the explicitly prepared image is missing"
                )
        except (OSError, subprocess.TimeoutExpired):
            errors.append("Docker image inspection failed")
    return {
        "ready": not errors,
        "harbor_version": version,
        "harbor_revision": HARBOR_REVISION,
        "image": image,
        "errors": errors,
    }


def write_evidence(directory: Path, raw: bytes) -> None:
    """Host-owned, exclusive evidence; never a writable task artifact mount."""
    if len(raw) > MAX_EVIDENCE_BYTES:
        raise HarborError("Evidence exceeds the declared response bound")
    directory.mkdir(mode=0o700)
    with (directory / "response.json").open("xb") as stream:
        stream.write(raw)
    with (directory / "manifest.json").open("x") as stream:
        json.dump(
            {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}, stream
        )


def read_evidence(directory: Path) -> bytes:
    """This reads our own smoke evidence, not an imported-archive admission API."""
    try:
        return _read_evidence(directory)
    except (OSError, ValueError) as error:
        raise HarborError(
            "Required evidence is missing or corrupt; ungradable"
        ) from error


def _read_evidence(directory: Path) -> bytes:
    if directory.is_symlink() or not directory.is_dir():
        raise HarborError("Missing or linked trusted evidence directory")
    contents = {}
    for name, maximum in (
        ("manifest.json", 1024),
        ("response.json", MAX_EVIDENCE_BYTES),
    ):
        descriptor = os.open(
            directory / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        )
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
                raise HarborError("Required evidence is not a bounded regular file")
            contents[name] = stream.read(maximum + 1)
            if len(contents[name]) > maximum:
                raise HarborError("Evidence grew past its bound during read")
    manifest = json.loads(contents["manifest.json"])
    raw = contents["response.json"]
    if manifest != {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}:
        raise HarborError("Required evidence is corrupt; ungradable")
    return raw


def run_smoke(output: Path, image: str = SMOKE_IMAGE) -> dict[str, Any]:
    """Run only the built-in fake endpoint; never contact a configured model."""
    ready = preflight(image)
    if not ready["ready"]:
        raise HarborError("; ".join(ready["errors"]))
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise HarborError(
            "Output must be a new directory; originals are never overwritten"
        )
    output.mkdir(mode=0o700)
    # Set before importing Harbor. No hub API or ambient model route is used.
    os.environ["HARBOR_TELEMETRY"] = "off"
    from .harbor_runtime import smoke

    return smoke(output, image)
