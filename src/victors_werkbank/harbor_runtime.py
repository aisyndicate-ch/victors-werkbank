"""Harbor-owned lifecycle for a bounded actual-M02 integration smoke.

The smoke runs M02's example instance, the one the committed task package
holds: it checks that Harbor, Docker and the image can run and regrade a trial,
and has no run whose instance value it could take (`harbor_controls`).

Imported only after the explicit Harbor preflight. No production model API.
"""

import asyncio
import hashlib
import json
import tempfile
import threading
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent  # type: ignore[import-untyped]
from harbor.models.trial.config import (  # type: ignore[import-untyped]
    SourceTrialConfig,
    TrialConfig,
)
from harbor.trial.trial import Trial  # type: ignore[import-untyped]
from harbor.verifier.verifier import Verifier  # type: ignore[import-untyped]

from .harbor_controls import GOLD, PROMPT, WRONG, grade_smoke_response
from .harbor_smoke import (
    GRADER_MODULES,
    MAX_EVIDENCE_BYTES,
    HarborError,
    read_evidence,
    write_evidence,
)

_GRADER = """import json, sys
from pathlib import Path
sys.path.insert(0, '/tests/domain')
from victors_werkbank.harbor_controls import grade_smoke_response
checks = grade_smoke_response(Path('/trusted/response.json').read_bytes())
Path('/logs/verifier/checks.json').write_text(json.dumps(checks))
Path('/logs/verifier/reward.txt').write_text('1' if all(checks.values()) else '0')
"""


class SmokeAgent(BaseAgent):
    """Fixed actual prompt and loopback transport; no model or tool execution."""

    @staticmethod
    def name() -> str:
        return "werkbank-harbor-smoke"

    def version(self) -> str:
        return "1"

    def __init__(self, *, endpoint: str, trusted: str, **kwargs: Any):
        super().__init__(**kwargs)
        url = urllib.parse.urlsplit(endpoint)
        if (
            url.scheme != "http"
            or url.hostname != "127.0.0.1"
            or url.username
            or url.password
        ):
            raise HarborError("Smoke endpoint must be the local fake HTTP server")
        self.endpoint = endpoint
        self.trusted = Path(trusted)

    async def setup(self, environment: Any) -> None:
        result = await environment.exec(
            command="test ! -e /tests && test ! -e /trusted "
            "&& test ! -e /var/run/docker.sock && test ! -e /run/docker.sock"
        )
        if result.return_code:
            raise HarborError("Private grader inputs leaked to the task environment")

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        if instruction.strip() != PROMPT.strip():
            raise HarborError("M02 instruction changed")
        payload = json.dumps(
            {
                "messages": [{"role": "user", "content": PROMPT}],
                "tools": [],
            }
        ).encode()

        def request() -> bytes:
            # No host proxy or redirects, even for the built-in loopback endpoint.
            class NoRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, *args: Any, **kwargs: Any) -> None:
                    raise HarborError("Smoke endpoint redirect rejected")

            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}), NoRedirect()
            )
            req = urllib.request.Request(
                self.endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with opener.open(req, timeout=5) as response:
                raw = response.read(MAX_EVIDENCE_BYTES + 1)
            if len(raw) > MAX_EVIDENCE_BYTES:
                raise HarborError("Oversized fake response")
            return raw

        raw = await asyncio.to_thread(request)
        write_evidence(self.trusted, raw)


class EvidenceVerifier(Verifier):
    """Only the host-owned raw response is authoritative for this tools-free case."""

    def __init__(self, *, trusted: str, **kwargs: Any):
        super().__init__(**kwargs)
        self.trusted = Path(trusted)

    async def verify(self) -> Any:
        raw = read_evidence(self.trusted)
        setup = await self.environment.exec(
            command="mkdir -p /trusted && chmod 700 /trusted"
        )
        if setup.return_code:
            raise HarborError("Trusted verifier directory setup failed")
        # Upload exactly the validated bytes, not a second read of the source.
        with tempfile.TemporaryDirectory(prefix="werkbank-verified-") as temporary:
            snapshot = Path(temporary) / "response.json"
            snapshot.write_bytes(raw)
            await self.environment.upload_file(snapshot, "/trusted/response.json")
        return await super().verify()


def _task(root: Path, image: str) -> Path:
    task = root / "task"
    (task / "tests" / "domain" / "victors_werkbank").mkdir(parents=True)
    (task / "environment").mkdir()
    (task / "instruction.md").write_text(PROMPT)
    (task / "task.toml").write_text(
        'schema_version = "1.4"\n[task]\nname = "werkbank/m02-smoke"\n'
        'version = "1.0.0"\n[environment]\n'
        f'docker_image = "{image}"\ncpus = 1\nmemory_mb = 512\ngpus = 0\n'
        'network_mode = "no-network"\n[agent]\ntimeout_sec = 15\n'
        '[verifier]\nenvironment_mode = "separate"\ntimeout_sec = 15\n'
    )
    package = Path(__file__).parent
    for name in GRADER_MODULES:
        target = task / "tests" / "domain" / "victors_werkbank" / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes((package / name).read_bytes())
    (task / "tests" / "grader.py").write_text(_GRADER)
    (task / "tests" / "test.sh").write_text(
        "#!/bin/sh\nset -eu\npython3 /tests/grader.py\n"
    )
    (task / "tests" / "test.sh").chmod(0o755)
    return task


def _hashes(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


async def _run(
    output: Path, image: str, endpoint: str, requests: list[str]
) -> dict[str, Any]:
    task = _task(output, image)
    rows = []
    configs = []
    for name, expected in (("correct", 1), ("wrong", 0), ("forbidden-tool", 0)):
        cfg = TrialConfig.model_validate(
            {
                "task": {"path": task},
                "trial_name": name,
                "trials_dir": output / "trials",
                "agent": {
                    "import_path": "victors_werkbank.harbor_runtime:SmokeAgent",
                    "kwargs": {
                        "endpoint": endpoint + "/" + name,
                        "trusted": str(output / name),
                    },
                },
                "verifier": {
                    "import_path": "victors_werkbank.harbor_runtime:EvidenceVerifier",
                    "kwargs": {"trusted": str(output / name)},
                },
            }
        )
        result = await (await Trial.create(cfg)).run()
        reward = result.verifier_result.rewards if result.verifier_result else None
        if result.exception_info or reward != {"reward": expected}:
            raise HarborError(
                f"Unexpected {name} outcome; inspect retained trial evidence"
            )
        checks = json.loads(
            (output / "trials" / name / "verifier" / "checks.json").read_text()
        )
        expected_failures = {
            "correct": set(),
            "wrong": {"v_solutions"},
            "forbidden-tool": {"no_tools"},
        }[name]
        if (
            {key for key, value in checks.items() if value is False}
            != expected_failures
            or not all(type(value) is bool for value in checks.values())
            or set(checks) != set(grade_smoke_response(b"{}"))
        ):
            raise HarborError(f"Unexpected individual {name} check predicates")
        rows.append({"control": name, "reward": reward, "checks": checks})
        configs.append(cfg)
    if requests != ["correct", "wrong", "forbidden-tool"]:
        raise HarborError("Unexpected endpoint requests or retry")
    return {"rows": rows, "configs": configs}


async def _regrade(output: Path, configs: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for cfg in configs:
        original = output / "trials" / cfg.trial_name
        before = _hashes(original)
        trusted_before = _hashes(Path(cfg.verifier.kwargs["trusted"]))
        regrade = cfg.model_copy(deep=True)
        regrade.source_trial = SourceTrialConfig(
            action="regrade", type="local", path=original
        )
        regrade.trial_name += "-regrade"
        result = await (await Trial.create(regrade)).run()
        reward = result.verifier_result.rewards if result.verifier_result else None
        expected = 1 if cfg.trial_name == "correct" else 0
        if result.exception_info or reward != {"reward": expected}:
            raise HarborError("Unexpected regrade outcome")
        if (
            _hashes(original) != before
            or _hashes(Path(cfg.verifier.kwargs["trusted"])) != trusted_before
        ):
            raise HarborError("Regrade changed original evidence")
        rows.append(
            {
                "control": regrade.trial_name,
                "reward": reward,
                "original_unchanged": True,
            }
        )
    return rows


def smoke(output: Path, image: str) -> dict[str, Any]:
    """Stop the server before native Harbor regrading; keep every original run."""
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            # Count every POST attempt, including requests rejected below.
            requests.append(self.path.removeprefix("/"))
            if self.path not in {"/correct", "/wrong", "/forbidden-tool"}:
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Invalid request length")
                body = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeError, RecursionError):
                self.send_error(400)
                return
            expected = {
                "messages": [{"role": "user", "content": PROMPT}],
                "tools": [],
            }
            if body != expected:
                self.send_error(400)
                return
            name = self.path.removeprefix("/")
            answer = WRONG if name == "wrong" else GOLD
            message: dict[str, Any] = {
                "role": "assistant",
                "content": json.dumps(answer),
            }
            reason = "stop"
            if name == "forbidden-tool":
                reason = "tool_calls"
                message["tool_calls"] = [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "calculate", "arguments": "{}"},
                    }
                ]
            raw = json.dumps(
                {"choices": [{"message": message, "finish_reason": reason}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = asyncio.run(
            _run(output, image, f"http://127.0.0.1:{server.server_port}", requests)
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    rows = result["rows"] + asyncio.run(_regrade(output, result["configs"]))
    report = {
        "schema": "werkbank.harbor-smoke.v1",
        "passed": True,
        "case": "M02",
        "requests": len(requests),
        "endpoint_stopped_before_regrade": True,
        "controls": rows,
        "comparison_eligible": False,
    }
    (output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
