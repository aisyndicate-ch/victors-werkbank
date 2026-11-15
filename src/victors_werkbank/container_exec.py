"""Code executors for Harbor containers (`container-exec-v2`), host side.

Both implement `domains.execution.Executor`, so case modules and trusted
helpers run unchanged. `ContainerExecutor` runs each execution in the
task container through two transport calls the Harbor agent supplies (an exec
and a file upload); the host holds the authoritative source bytes and sends a
fresh copy with every request, and reads back only the controller's bounded
result. `LocalExecutor` runs the same controller in the separate verifier
container, inside the grader. Imports no Harbor module.
"""

import base64
import binascii
import json
import math
import os
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import container_controller as controller
from .agent.transport import strict_json
from .domains.execution import MAX_OUTPUT, MAX_SOURCE_BYTES, Execution
from .isolation_errors import SandboxUnavailable

# Where the agent installs the controller in the task container.
INSTALL_DIR = "/opt/werkbank"
CONTROLLER = f"{INSTALL_DIR}/container_controller.py"
HELPERS_DIR = f"{INSTALL_DIR}/sandbox_helpers"
# Seconds a container exec may take beyond its execution's own deadline and
# cleanup; only a backstop, which ends the case as an infrastructure error.
EXEC_MARGIN_SECONDS = 30
INSTALL_TIMEOUT_SECONDS = 120
MAX_RESULT_BYTES = 4 * MAX_OUTPUT + 4096
REASONS = frozenset(
    {"completed", "compile_failed", "execution_failed", "timeout", "output_limit"}
)
# What the compile stage's probe finds in its own process; the order is the
# order of `sandbox_helpers/container_probe.rs`.
COMPILE_LIMITS = (
    controller.COMPILE_SECONDS,
    controller.COMPILE_FILE_SIZE,
    controller.COMPILE_PROCESSES,
    controller.OPEN_FILES,
    controller.COMPILE_ADDRESS_SPACE,
)
# exec(command, timeout_seconds) -> (return code, stdout); upload(file, target).
Exec = Callable[[str, int], tuple[int, str]]
Upload = Callable[[Path, str], None]


class ExecutorFailure(RuntimeError):
    """The container executor failed; an infrastructure error, never a result."""


def helper_files() -> list[Path]:
    """The trusted helpers installed next to the controller."""
    helpers = Path(__file__).with_name("sandbox_helpers")
    return sorted(
        path for path in helpers.iterdir() if path.suffix in controller.SOURCE_SUFFIXES
    )


def _source_files(source: Path) -> dict[str, str]:
    """Every regular file of the submission directory, base64; the case module
    has checked its names, links and sizes before it executes."""
    files = {}
    for path in sorted(source.iterdir()):
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_SOURCE_BYTES:
            raise ValueError("Submission contains unsupported artifacts")
        files[path.name] = base64.b64encode(path.read_bytes()).decode()
    return files


def request(source: Path, helper: str, data: bytes, timeout: float) -> dict[str, Any]:
    if (
        timeout <= 0
        or not math.isfinite(timeout)
        or len(data) > MAX_SOURCE_BYTES
        or helper not in controller.HELPERS
    ):
        raise ValueError("Invalid executor budget/input")
    return {
        "helper": helper,
        "timeout": float(timeout),
        "data": base64.b64encode(data).decode(),
        "files": {} if helper in controller.PROBES else _source_files(source),
    }


def execution(result: Any) -> Execution:
    """An `Execution` from the controller's result, checked strictly."""
    try:
        if not isinstance(result, dict) or set(result) != {
            "stdout",
            "stderr",
            "returncode",
            "reason",
            "elapsed_seconds",
        }:
            raise ValueError
        stdout = base64.b64decode(result["stdout"], validate=True)
        stderr = base64.b64decode(result["stderr"], validate=True)
        elapsed = result["elapsed_seconds"]
        if (
            len(stdout) > MAX_OUTPUT
            or len(stderr) > MAX_OUTPUT
            or type(result["returncode"]) is not int
            or result["reason"] not in REASONS
            or type(elapsed) is not float
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError
    except (ValueError, TypeError, binascii.Error):
        raise ExecutorFailure("Container executor returned an invalid result") from None
    return Execution(stdout, stderr, result["returncode"], result["reason"], elapsed)


def _probe(executor: Any, empty_source: Path) -> Execution:
    """Every code episode probes the executor and its compile stage, so an
    image without the toolchain is unsupported for each of them alike. The
    result is the Python probe's, the record the episode journals."""
    limits = " ".join(map(str, COMPILE_LIMITS)).encode()
    results = []
    for helper, data in (
        ("container_probe.py", b""),
        ("container_probe.rs", limits),
    ):
        try:
            result = executor.run(empty_source, helper, data, timeout=5.0)
        except ExecutorFailure:
            raise SandboxUnavailable("Container executor preflight failed") from None
        if result.reason != "completed" or b'"isolated": true' not in result.stdout:
            raise SandboxUnavailable(
                "Container executor preflight failed: "
                + result.stderr.decode("utf-8", errors="replace")[:2000]
            )
        results.append(result)
    return results[0]


class ContainerExecutor:
    """Executes in the task container through the agent's transport calls."""

    def __init__(self, exec_call: Exec, upload: Upload) -> None:
        self._exec = exec_call
        self._upload = upload
        self._count = 0

    def install(self) -> None:
        """Copy the controller and its helpers in and initialize them, as root."""
        with tempfile.TemporaryDirectory(prefix="werkbank-controller-") as temporary:
            staged = Path(temporary)
            (staged / "controller.py").write_bytes(
                Path(controller.__file__).read_bytes()
            )
            self._command(f"mkdir -p {HELPERS_DIR}", INSTALL_TIMEOUT_SECONDS)
            self._upload(staged / "controller.py", CONTROLLER)
            for path in helper_files():
                self._upload(path, f"{HELPERS_DIR}/{path.name}")
        output = self._command(
            f"chown -R 0:0 {INSTALL_DIR} && chmod -R a-w,a+rX {INSTALL_DIR} "
            f"&& {controller.PYTHON} -I {CONTROLLER} init {HELPERS_DIR}",
            INSTALL_TIMEOUT_SECONDS,
        )
        try:
            ready = strict_json(output.splitlines()[-1])
        except (ValueError, IndexError):
            ready = None
        if ready != {"ready": True, "semantics": controller.SEMANTICS}:
            raise SandboxUnavailable("Container executor installation failed")

    def _command(self, command: str, timeout: int) -> str:
        try:
            code, output = self._exec(command, timeout)
        except Exception:
            raise ExecutorFailure("Container executor transport failed") from None
        if code != 0:
            raise ExecutorFailure("Container executor failed")
        return output

    def run(
        self, source: Path, helper: str, data: bytes = b"", *, timeout: float = 5.0
    ) -> Execution:
        payload = json.dumps(request(source, helper, data, timeout)).encode()
        self._count += 1
        target = f"{controller.SPOOL}/request-{self._count}.json"
        fd, name = tempfile.mkstemp(prefix="werkbank-request-")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
            try:
                self._upload(Path(name), target)
            except Exception:
                raise ExecutorFailure("Container executor transport failed") from None
        finally:
            os.unlink(name)
        build = controller.COMPILE_SECONDS if helper in controller.COMPILED else 0
        backstop = math.ceil(timeout) + build + EXEC_MARGIN_SECONDS
        output = self._command(
            f"{controller.PYTHON} -I {CONTROLLER} run {target}", backstop
        )
        if len(output) > MAX_RESULT_BYTES:
            raise ExecutorFailure("Container executor returned an invalid result")
        try:
            reply = strict_json(output)
        except ValueError:
            raise ExecutorFailure(
                "Container executor returned an invalid result"
            ) from None
        if not isinstance(reply, dict) or set(reply) != {"result"}:
            raise ExecutorFailure("Container executor failed")
        return execution(reply["result"])

    def probe(self, empty_source: Path) -> Execution:
        return _probe(self, empty_source)


class LocalExecutor:
    """Executes in the current container through the controller module, as the
    verifier's grader does; `init` must have installed the helpers. Each
    execution wipes `/tmp`, so the caller keeps its own files elsewhere."""

    def run(
        self, source: Path, helper: str, data: bytes = b"", *, timeout: float = 5.0
    ) -> Execution:
        try:
            result = controller.execute(request(source, helper, data, timeout))
        except controller.ControllerError:
            raise ExecutorFailure("Container executor failed") from None
        return execution(result)

    def probe(self, empty_source: Path) -> Execution:
        return _probe(self, empty_source)
