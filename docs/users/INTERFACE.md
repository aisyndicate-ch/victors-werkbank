# Browser interface

`werkbank-ui` is Werkbank in the browser, on this computer only: the guided start
of [`werkbank start`](CLI.md#guided-start) with its questions as forms, the
running series with a stop button, the result page, the offer to submit, and a
view of run and evaluation directories. It is dark throughout and loads nothing
from the internet. Its one script, served by the interface itself, keeps a
running series page current and does nothing else. With `--read-only` it is the
view alone: it starts nothing and writes nothing.

## Start

```sh
./werkbank --browser            # installs on the first call, then serves the interface
.venv/bin/werkbank start --browser
.venv/bin/werkbank-ui           # the same, without flags for the start
.venv/bin/werkbank-ui --read-only runs/
```

The command prints an address with this session's token, for example
`http://127.0.0.1:41733/?token=…`, and serves until Ctrl-C. Open it in a browser on
the same machine; under WSL2 the Windows browser reaches it as well, as
`127.0.0.1` or `localhost` ([platforms](PLATFORMS.md#wsl2)), and in a Lima VM on a
Mac the Mac's browser opens the printed address, as Lima forwards the port
([macOS](PLATFORMS.md#macos-experimental)). Nothing opens a browser for you.

- `werkbank start --browser` takes the flags of `werkbank start` (not `--yes`):
  they become the defaults of the form, and what the form does not offer
  (`--mode`, `--passes`, `--output`, `--serving-note`, `--image`, `--chat-template-kwargs`)
  applies as given. New series go under `runs/`.
- `werkbank-ui [DIRECTORY ...]` shows the run and evaluation directories under its
  arguments (default `runs`); new series go under the first one, which is created
  when missing. `--port N` fixes the port, otherwise a free one is chosen.
- A series runs in the process that serves the interface. Closing the browser
  changes nothing, and the page can be opened again. Ending the process (Ctrl-C, a
  closed terminal, a termination) first stops the running pass, as Ctrl-C stops a
  `werkbank run`, and then ends with exit `0`. Exit `2` means it could not serve.

## A series

**New series** is a form with the server address and the model, both optional, and
under *Settings* the hardware label, the per-response cap, temperature, top-p,
top-k, thinking and streaming. Each filled field is the flag of `werkbank start`
it stands for and passes that command's own checks; a refused form starts nothing
and says why. *Begin* then runs the steps of the guided start, with the same
texts as at the terminal:

1. **Readiness, server, model, server check, mode.** Without an address, Werkbank
   sends one read-only model list request to each usual port on `127.0.0.1` and to
   no other host; a request to the model goes out only after your choice, or with
   exactly one server and one model found. Whatever the start would ask at a
   terminal (several servers or models, the mode, three passes or one for Quick,
   Normal and Full, the confirmation to start)
   appears as one pending question, under the lines the start said before it.
2. **Passes.** Each pass is a `werkbank run` child process, as at the terminal,
   and draws its own [instance value](READING_RESULTS.md#instance-values).
   The page shows, per pass, the episodes the run has reported with their status
   and reason, and the log. It stays current by itself: every five seconds the
   interface's script reads the page again and puts in what changed, and the log
   follows new lines only while you are at its end. With scripts switched off in
   the browser, the page loads itself again every five seconds instead. This
   display follows the run's progress lines; the run's records, under *Runs and
   evaluations*, are what counts.
3. **Evaluation and result.** The evaluation and the result page are written into
   the series folder as `werkbank start` writes them, and the page is shown inside
   the interface.
4. **Submission.** After the three passes of Quick, Normal, Full or German the start offers to
   submit, as at the terminal; a trial or a single pass is never offered, and its
   series page shows no submission step. See [submitting](#submitting).

**Stop** ends the series. While a pass runs, it interrupts that pass once, as
Ctrl-C does, and the run ends by its own rules, which can take a minute while its
containers stop; the pass's folder stays as evidence, and the log names the
commands that finish the series by hand. At a question or between steps, the
start ends there and nothing more is started. There is no pause and no resume: a
stopped pass is run again as a whole. A technically failed pass is offered once
for a repeat, as at the terminal, and the result page names the repeat.

Only one series runs at a time. *Close this series* forgets an ended one, whose
folder stays, and offers the form again.

## Submitting

The offer and everything after it is the submit flow of
[`werkbank submit`](../SUBMISSION.md), with its dialog in the browser: the five
optional questions, then the summary of exactly what would be sent and to which
address, the consent text, the separate question for raw data and the archive's
size and digest, all on the page above the final question *Send now?*. Only the
*Yes* of that pending question sends, once. *No*, *Stop*, a closed browser or an
answer to an earlier question send nothing. The privacy check and every refusal
of `werkbank submit` apply unchanged.

## What the view shows

- **Directories:** every run directory (it holds `plan.json`) and evaluation
  directory (`evaluation.json`) under the arguments, up to eight levels deep. Links
  are not followed and hidden directories are skipped. The list shows each run's
  model, cases, whether this build wrote it and whether `run.json` exists, read
  from its files as written.
- **A run:** the run's state under this build, then its report, which names the
  run's [instance value](READING_RESULTS.md#instance-values) and whether it was
  drawn or given:
  - *complete*: this build validated the run as `werkbank validate` does, and the
    report is rendered from the validated records exactly as `werkbank report`
    renders it, in memory only;
  - *interrupted*: this build validated the episodes recorded so far (no
    `run.json`; interrupted, or still running), listed with their status and
    reason;
  - *another build*: the run's harness digest is not this build's, so this build
    cannot validate it; the `report.md` the run's build wrote is shown as written
    and labelled unchecked;
  - *invalid*: validation failed; only its reason is shown, never a stored
    report.
- **An evaluation:** checked as `werkbank validate EVAL` checks it without
  sources (shape, consistency, every file's digest, and the renderings when this
  build wrote it), then `evaluation.md`, the model card, the dataset card and every
  CSV table, with each pool's instance values beside its seeds as those files hold
  them. A file is shown only while it still has the digest the record binds.
- **A result page:** the `result` directory beside an evaluation, as a series has
  it, only while it is exactly what this build renders for that evaluation. It is
  shown in its own dark scheme whatever the browser prefers; the file on disk is
  never changed.

A page's validation is kept only until any file under the directory changes
(Harbor's `trials/` aside), so a run that is still being written, or one damaged
after it was viewed, shows its current state.

Never shown: journals, reasoning, raw responses, code snapshots, Harbor's `trials/`
and anything else a page does not name above. Every value is escaped: the Markdown
that Werkbank writes becomes headings, lists, tables and code spans, and nothing
else in it is ever markup. A series page shows the lines a terminal would show,
which include the server address you gave or the search found. That address
lives in the serving process only; the interface writes no file holding it.

## Security

The interface is local only and stores no credentials: it has no field for an API
key. A key comes only from the environment variable `WERKBANK_API_KEY` of the
process that started the interface, and it goes only to a server you named in the
form's endpoint field, never to one the search found
([API key](CLI.md#api-key)). Its controls:

- It listens on `127.0.0.1` only; there is no option for another address.
- A per-session token (32 random bytes) is printed with the address. Its first
  use sets a cookie (`HttpOnly`, `SameSite=Strict`, named after the port) and
  redirects to the start page; every other request needs that cookie.
- The `Host` header must be `127.0.0.1` or `localhost` with the interface's port,
  which defeats DNS rebinding. A request with a foreign `Origin` or with
  `Sec-Fetch-Site` other than `same-origin` or `none` is refused, so no other
  website in the same browser can read or trigger anything.
- Reading takes `GET` and `HEAD`. Whatever acts (begin, answer, stop, close) is a
  `POST` of a form, and it must carry this interface's `Origin`, the cookie and a
  second random token that only this session's forms hold; a form is at most
  16 KiB. An answer also names the pending question by a random ID of its own, so
  an old or repeated form, or one for another question, changes nothing. With
  `--read-only` no `POST` is answered.
- Every HTTP/1.x answer, the standard library's own error answers included,
  carries a content security policy that loads nothing but the interface's own
  stylesheet and, with the control side, its own script (no inline script and
  none from elsewhere; the script may read this interface's pages and sends no
  form; forms and frames only to the interface itself; with `--read-only` no
  script, no forms and no frames), `nosniff`, the referrer policy
  `same-origin` (under which a form names this interface's origin and nothing goes
  to another site) and `no-store`.
  A result page is served with a policy of its own: its inline style and
  graphics, no scripts, no forms, sandboxed, framed by the interface only.
- Pages address directories by opaque keys from the scan, never by paths from the
  request; files are read bounded, as regular files and never through a link. The
  request log carries neither the token nor control characters, and never a form's
  fields.
- A form can set only the fields the page shows, each as one value of its flag,
  never a further flag; no form names a path, a command or the receiver.

The token keeps other users and programs on the same machine from reading the
interface or acting through its port. Browsers do not separate cookies by port, so
the browser also sends the cookie to any other server on `127.0.0.1` the user opens
in it, as with Jupyter's token; the form token is never sent that way. Whoever can
read the terminal or run code as the same user can read the run directories and
start runs directly.
