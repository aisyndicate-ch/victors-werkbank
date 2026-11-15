"""The loopback HTTP server of the interface.

It binds to 127.0.0.1 only. Every request must name this server in its Host
header (127.0.0.1 or localhost with its port, against DNS rebinding), carry
no foreign Origin or cross-site fetch metadata, and present the per-session
token: once in the address the command prints, which turns it into a
same-site, HTTP-only cookie, then as that cookie.

Read-only, it answers GET and HEAD only. With the control side it also takes
the forms that begin, answer, stop and close a series: a POST must carry this
server's Origin, the cookie and the form token every form of this session
holds, so no other page and no other program can start or answer anything."""

import hmac
import secrets
import sys
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TextIO
from urllib.parse import parse_qs, urlsplit

from . import control_pages
from .control import Busy, Control, Stale
from .pages import Pages, checked_result, page
from .script import LIVE
from .style import STYLE

LOOPBACK = "127.0.0.1"
POLICY = (
    "default-src 'none'; style-src 'self'; base-uri 'none'; form-action 'none';"
    " frame-ancestors 'none'"
)
# With the control side: forms go to this server only, a page may frame this
# server's own result page, and the one script this server serves may read
# this server's pages, which keeps a running series page current in place. No
# inline script and none from elsewhere.
CONTROL_POLICY = (
    "default-src 'none'; style-src 'self'; script-src 'self'; connect-src 'self';"
    " base-uri 'none'; form-action 'self'; frame-src 'self'; frame-ancestors 'none'"
)
# A result page: its own inline style and graphics and nothing else, no
# scripts, no forms, framed by this server only.
RESULT_POLICY = (
    "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none';"
    " form-action 'none'; frame-ancestors 'self'; sandbox"
)
# The referrer policy is same-origin, not no-referrer: under no-referrer a
# browser sends `Origin: null` with a form's POST, and every form of the
# interface would be refused as of a foreign origin. With same-origin a form
# names the interface's own origin, and nothing goes to another site.
SECURITY_HEADERS = (
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "same-origin"),
    ("Cache-Control", "no-store"),
    ("Cross-Origin-Opener-Policy", "same-origin"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
)
FETCH_SITES = frozenset({"same-origin", "none"})
HTML = "text/html; charset=utf-8"
FORM = "application/x-www-form-urlencoded"
MAX_FORM_BYTES = 16 * 1024
RESULT = "result"


class Viewer(ThreadingHTTPServer):
    """One viewer session: its roots, its port and its token."""

    daemon_threads = True

    def __init__(
        self, roots: list[Path], port: int = 0, control: Control | None = None
    ) -> None:
        super().__init__((LOOPBACK, port), Handler)
        self.control = control
        self.port = self.server_address[1]
        self.token = secrets.token_urlsafe(32)
        self.hosts = frozenset({f"127.0.0.1:{self.port}", f"localhost:{self.port}"})
        self.origins = frozenset(f"http://{host}" for host in self.hosts)
        self.cookie = f"werkbank_ui_{self.port}"
        self.pages = Pages(roots, control=control is not None)
        self.log: TextIO = sys.stderr

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/?token={self.token}"


class Handler(BaseHTTPRequestHandler):
    server: Viewer
    server_version = "werkbank-ui"
    sys_version = ""
    timeout = 30

    def do_GET(self) -> None:
        self._serve(body=True)

    def do_HEAD(self) -> None:
        self._serve(body=False)

    def _page(self, title: str, body: str) -> str:
        return page(title, body, control=self.server.control is not None)

    def _refuse_method(self) -> None:
        if self.server.control is None:
            text, allowed = "This view is read-only.", "GET, HEAD"
        else:
            text, allowed = "This method is not answered.", "GET, HEAD, POST"
        self._send(
            405,
            self._page("Not allowed", f"<p>{text}</p>"),
            headers=[("Allow", allowed)],
        )

    do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _refuse_method
    do_TRACE = do_CONNECT = _refuse_method

    def do_POST(self) -> None:
        if self.server.control is None:
            self._refuse_method()
            return
        # Whatever is refused, the connection ends: an unread body must never
        # be taken for the next request.
        self.close_connection = True
        refusal = self._refusal()
        origins = self.headers.get_all("Origin") or []
        if refusal is None and len(origins) != 1:
            refusal = "No origin"
        if refusal is None and not self._has_cookie():
            refusal = "Not signed in"
        if refusal is not None:
            self._send(403, self._page("Refused", f"<p>{refusal}.</p>"))
            return
        form = self._form()
        if form is None:
            self._send(400, self._page("Refused", "<p>Not a form of this page.</p>"))
            return
        tokens = form.get("form", [])
        if len(tokens) != 1 or not hmac.compare_digest(
            tokens[0].encode(), self.server.control.form_token.encode()
        ):
            self._send(403, self._page("Refused", "<p>Wrong form token.</p>"))
            return
        try:
            self._act(urlsplit(self.path).path, form)
        except Exception as error:  # Never a traceback to the browser.
            self.log_message("%s", type(error).__name__)
            self._send(500, self._page("Error", "<p>Internal error.</p>"))

    def _form(self) -> dict[str, list[str]] | None:
        """The fields of a URL-encoded form of bounded size, or None."""
        kind = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
        length = self.headers.get("Content-Length", "")
        if kind != FORM or not (length.isascii() and length.isdigit()):
            return None
        size = int(length)
        if size > MAX_FORM_BYTES:
            return None
        try:
            text = self.rfile.read(size).decode("utf-8")
            return parse_qs(text, keep_blank_values=True, strict_parsing=bool(text))
        except ValueError:
            return None

    def _act(self, path: str, form: dict[str, list[str]]) -> None:
        """One action of the control side; each answers with a redirect to the
        series page, so a reloaded page never repeats it."""
        control = self.server.control
        assert control is not None

        def one(name: str) -> str:
            values = form.get(name, [])
            return values[0] if len(values) == 1 else ""

        session = control.session
        if path == "/series/start":
            from werkbank_start.cli import form_options

            fields = {name: one(name) for name in form if name != "form"}
            try:
                options = form_options(control.defaults, fields)
                control.begin(options)
            except ValueError as error:
                self._send(
                    400,
                    control_pages.new_series(
                        control.form_token, control.defaults, str(error)
                    ),
                )
                return
            except Busy:
                pass
        elif path == "/series/answer" and session is not None:
            try:
                session.answer(one("question"), one("value"))
            except Stale:
                pass
        elif path == "/series/stop" and session is not None:
            session.stop()
        elif path == "/series/close":
            control.close()
        else:
            self._send(404, self._page("Not found", "<p>Not found.</p>"))
            return
        self._send(
            303,
            self._page("Done", '<p><a href="/series">Continue</a></p>'),
            headers=[("Location", "/series")],
        )

    def send_error(
        self, code: int, message: str | None = None, explain: str | None = None
    ) -> None:
        # The standard library's own errors (a malformed request line, an
        # unknown method) carry the security headers too, and a fixed body.
        self.close_connection = True
        body = self.command != "HEAD" and code >= 200 and code not in {204, 304}
        self._send(code, self._page("Error", f"<p>Error {code}.</p>"), body=body)

    def _refusal(self) -> str | None:
        """Why the request is refused before any routing, or None."""
        hosts = self.headers.get_all("Host") or []
        if len(hosts) != 1 or hosts[0] not in self.server.hosts:
            return "Unknown host"
        origins = self.headers.get_all("Origin") or []
        if any(origin not in self.server.origins for origin in origins):
            return "Foreign origin"
        sites = self.headers.get_all("Sec-Fetch-Site") or []
        if any(site not in FETCH_SITES for site in sites):
            return "Cross-site request"
        return None

    def _token(self, value: str) -> bool:
        return hmac.compare_digest(value.encode(), self.server.token.encode())

    def _has_cookie(self) -> bool:
        try:
            jar = SimpleCookie(self.headers.get("Cookie", ""))
        except CookieError:
            return False
        morsel = jar.get(self.server.cookie)
        return morsel is not None and self._token(morsel.value)

    def _serve(self, *, body: bool) -> None:
        refusal = self._refusal()
        if refusal is not None:
            self._send(403, self._page("Refused", f"<p>{refusal}.</p>"), body=body)
            return
        url = urlsplit(self.path)
        query = parse_qs(url.query, keep_blank_values=True)
        if "token" in query:
            tokens = query["token"]
            if len(tokens) != 1 or not self._token(tokens[0]):
                self._send(403, self._page("Refused", "<p>Wrong token.</p>"), body=body)
                return
            cookie = f"{self.server.cookie}={self.server.token}"
            home = "/" if self.server.control is None else "/series"
            self._send(
                303,
                self._page("Signed in", f'<p><a href="{home}">Continue</a></p>'),
                body=body,
                headers=[
                    ("Location", home),
                    ("Set-Cookie", f"{cookie}; Path=/; HttpOnly; SameSite=Strict"),
                ],
            )
            return
        if not self._has_cookie():
            self._send(
                403,
                self._page(
                    "Refused",
                    "<p>Open the address the <code>werkbank-ui</code> command printed"
                    " in its terminal.</p>",
                ),
                body=body,
            )
            return
        try:
            found = self._route(url.path)
        except Exception as error:  # Never a traceback to the browser.
            self.log_message("%s", type(error).__name__)
            self._send(500, self._page("Error", "<p>Internal error.</p>"), body=body)
            return
        if found is None:
            self._send(404, self._page("Not found", "<p>Not found.</p>"), body=body)
        elif isinstance(found, tuple) and found[0] == RESULT:
            self._send(200, found[1], body=body, policy=RESULT_POLICY)
        elif isinstance(found, tuple):
            self._send(200, found[1], body=body, content_type=found[0])
        else:
            self._send(200, found, body=body)

    def _route(self, path: str) -> str | tuple[str, str] | None:
        pages = self.server.pages
        if path == "/":
            return pages.index()
        if path == "/static/style.css":
            return ("text/css; charset=utf-8", STYLE)
        control = self.server.control
        if control is not None and path == "/static/live.js":
            return ("text/javascript; charset=utf-8", LIVE)
        if control is not None and path == "/series":
            session = control.session
            if session is None:
                return control_pages.new_series(control.form_token, control.defaults)
            return control_pages.series(session.snapshot(), control.form_token)
        if control is not None and path == "/series/result":
            session = control.session
            found = session.snapshot() if session is not None else None
            if found is None or found.evaluation is None or found.result is None:
                return None
            document = checked_result(found.evaluation, found.result)
            return None if document is None else (RESULT, document)
        parts = path.split("/")[1:]
        if len(parts) >= 2 and parts[0] in {"run", "evaluation"}:
            entry = pages.entry(parts[1], parts[0])
            if entry is None:
                return None
            if parts[0] == "run" and len(parts) == 2:
                return pages.run(entry)
            if parts[0] == "evaluation":
                if len(parts) == 2:
                    return pages.evaluation(entry)
                if len(parts) == 3 and parts[2] == RESULT:
                    document = pages.result(entry)
                    return None if document is None else (RESULT, document)
                if len(parts) == 3:
                    return pages.document(entry, parts[2])
                if len(parts) == 4 and parts[2] == "table":
                    return pages.table(entry, parts[3])
        return None

    def _send(
        self,
        status: int,
        content: str,
        *,
        body: bool = True,
        content_type: str = HTML,
        headers: list[tuple[str, str]] | None = None,
        policy: str | None = None,
    ) -> None:
        data = content.encode("utf-8")
        self.send_response(status)
        if policy is None:
            policy = POLICY if self.server.control is None else CONTROL_POLICY
        framing = "SAMEORIGIN" if policy == RESULT_POLICY else "DENY"
        for name, value in [
            ("Content-Security-Policy", policy),
            ("X-Frame-Options", framing),
            *SECURITY_HEADERS,
            *(headers or []),
        ]:
            self.send_header(name, value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if body:
            self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        # The request line can carry the token and control characters; the
        # log carries neither.
        message = "".join(
            ch if ch.isprintable() else f"\\x{ord(ch):02x}"
            for ch in (format % args).replace(self.server.token, "<token>")
        )
        self.server.log.write(f"werkbank-ui: {message}\n")
