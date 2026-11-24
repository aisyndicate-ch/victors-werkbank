# Submitting results

A submission sends the results of one evaluation to the Werkbank maintainer: one
archive, sent once over HTTPS, after you have seen exactly what it holds and
confirmed it. Nothing is sent automatically, and no other part of Werkbank contacts
the maintainer.

This build sends to `https://submit.victorswerkbank.workers.dev/submissions` and
names `werkbank-loeschung.jeep654@passinbox.com` as the contact for deletion
requests. The summary shows both before you confirm.

## Command

```text
werkbank submit EVALUATION --result DIR [--source RUN ...] [--write FILE]
```

- `EVALUATION` is an evaluation directory that this build wrote with
  `werkbank compare`.
- `--result DIR` is its result page directory, exactly as `werkbank result` wrote
  it for that evaluation with this build, beside the evaluation directory. A page
  of another evaluation, an edited page or a directory with another file is
  refused.
- `--source RUN` gives the evaluated runs, once each and in any order. With them
  the evaluation is checked completely and raw data can be offered. A run that the
  evaluation does not hold, a run given twice or a missing run is refused. Without
  `--source`, raw data is not offered.
- `--write FILE` writes the exact archive to this new file instead of sending it
  ([verifying what is sent](#verifying-what-is-sent)).

Only a `y` typed at a terminal confirms. When the input or the output is not a
terminal, the command refuses before the questions, and no option confirms.

A trial of `werkbank start` and a single pass of Quick, Normal, Full or German are never
submitted:
`werkbank submit` refuses a result page marked as one, and an evaluation of exactly
the trial's tasks.

An evaluation of a module's tasks, outside the score, is submitted like any other
of three passes; it holds no points, only the tasks solved and the subgoal
fulfilment.

`werkbank start` offers this same flow at the end of a series of three passes, at a
terminal and never with `--yes`; with `--yes` it prints the command instead
([guided start](users/CLI.md#guided-start)). In the
[browser interface](users/INTERFACE.md#submitting) the flow runs on the series
page, and the *Yes* to its final question takes the place of the typed `y`.

| Exit code | Meaning |
| --- | --- |
| `0` | The receiver confirmed the receipt, or the file was written. |
| `2` | Refused before the confirmation; something needs fixing. |
| `3` | Nothing was confirmed: you declined, the input ended, you pressed Ctrl-C, the receiver gave no valid receipt, or the file or the installation ID could not be written. |

## What is sent

- **Always:**
  - the result page: `index.html` and `result.json`;
  - the evaluation directory as `werkbank compare` wrote it: the task list, the
    scores and tables, and for each pool the served model name, the serving
    settings the evaluation records (such as engine and version, context length
    and quantization), the run configuration, the hardware statement (your
    hardware label, verbatim; otherwise the serving note's accelerator or the
    detected GPU product names), the container image's digest, without its
    registry host or name, and each run's ID, seed and
    [instance value](users/READING_RESULTS.md#instance-values), the random value
    its task data were built from. It never holds the server's model listing
    or a model path;
  - the envelope `submission.json`: the Werkbank version and build digest, the case
    set, the evaluated runs by ID and digest, your answers to the five questions,
    your answer to the [leaderboard](#leaderboard) question with your display
    name, and the installation ID.
- **Only with a separate yes:** raw data, the unchanged `werkbank export` output
  of every evaluated run. It holds every model response with its reasoning, every
  tool call and result, the task files as the run used them (with the task text
  of that run's instance), the run records
  and the server's model listing, in which a local model location appears only as
  its last part and a hash. Raw data covers every evaluated run or none.
- **Never:** the endpoint address, host names, user names, file paths, API keys or
  other credentials, Harbor's trial files or a run's `report.md`. An API key set
  in `WERKBANK_API_KEY` is never part of a run, an evaluation or a result page,
  so it is never part of a submission, and the privacy check looks for it all the
  same.

A submission is your own report. It is never ranked and never published unless
you agree to the [leaderboard](#leaderboard), a question of its own that comes only
with raw data.

## Steps before sending

The command works through these steps in order and sends nothing when one fails:

1. It asks the five [questions](#questions), because the answers are part of the
   archive.
2. It checks the evaluation as `werkbank validate` does, that this build wrote it,
   that every pool was evaluated on the same tasks, that the result page belongs
   to it and, with `--source`, that each run's export matches the digests the
   evaluation records. Then it builds the archive within its
   [limits](#the-archive) and runs the [privacy check](#privacy-check).
3. It shows a summary: the receiver address and the deletion contact; for each
   pool, and for each entry of runs with a given instance value, which the
   summary names as one and numbers apart from the pools, the served model name,
   the serving settings, the run configuration, the hardware statement with its
   source and the container image digest; the case
   set, the runs, the Werkbank version and the short build digest; every file with
   its size; the compressed size with and without raw data, or why raw data cannot
   be included; and the installation ID.
4. It shows the [consent text](#consent-text) and asks `Include raw data? [y/N]`
   (not asked when raw data cannot be included).
5. With raw data and the Full set it shows the
   [leaderboard's consent text](#leaderboard) and asks
   `Publish on the leaderboard? [y/N]` and, after a `y`,
   `Display name (Enter for anonymous):`. It prints your answer
   (`Leaderboard: publish yes` with `Display name:`, or `Leaderboard: publish no`;
   without raw data or for another case set, that the results are not published).
6. It prints the archive's size and SHA-256 and asks `Send now? [y/N]`, or
   `Write the file? [y/N]` with `--write`.

Only an explicit `y` counts. Any other answer, Ctrl-C or the end of input sends
nothing.

## Questions

Five multiple-choice questions in English. Each can be skipped with Enter, and a
skipped one stays empty. There is no free text.

| Field | Question | Choices |
| --- | --- | --- |
| `purpose` | What did you run Werkbank for? | `choose_model` Choosing a model; `compare_setup` Comparing serving setups (engine, quantization, hardware); `check_server` Checking my server; `curiosity` Curiosity; `other` Something else |
| `setup` | How did installing and starting go? | `smooth` Smoothly; `minor_problems` Minor problems; `major_problems` Major problems |
| `duration` | How was the run time? | `fine` Fine; `long_but_ok` Long but acceptable; `too_long` Too long |
| `clarity` | How clear is the result page? | `clear` Clear; `partly` Partly clear; `unclear` Unclear |
| `plausibility` | Do the results match your own experience with this model? | `yes` Yes; `partly` Partly; `no` No; `no_experience` I have no experience with it |

## Consent text

The command shows this text after the summary and before the prompts, verbatim:

```text
This sends your results to the Werkbank maintainer at the address shown above.

Always sent:
- the result page and the evaluation tables of this comparison
- the Werkbank version and its build digest, and the case set
- the served model name, the serving settings Werkbank recorded, the
  hardware statement (your hardware label as you wrote it, or the
  detected GPU product names) and the container image digest, listed
  with their values in the summary above
- your answers to the questions (skipped ones stay empty)
- a random installation ID, stored on this computer, that identifies
  your submissions without an account

Never sent: your endpoint address, host names, user names, file paths,
API keys or other credentials.

Raw data is sent only if you answer yes to "Include raw data?". It is
each run's export archive: every model response including its
reasoning, every tool call and result, the task files, the run records
and the server's model listing, in which a local model location appears
only as its last part and a hash.

The maintainer uses submissions to improve Werkbank and its cases. A
submission is your own report: it is never ranked and never published
unless you also answer yes to "Publish on the leaderboard?", which is
asked only with raw data. It is kept until you ask for deletion, at most
12 months after it arrives. To have every submission under your
installation ID deleted, write to the maintainer at the contact shown
above and quote the ID. A deleted submission is gone from the
receiver's backups within 30 days.

The receiver sees your IP address, as every web server does. It never
stores it with your submission; its server logs keep it for at most
7 days, only to limit abuse.
```

## Leaderboard

The maintainer's leaderboard shows setups (model, quantization, engine, hardware
and settings) by hardware class. A submission can enter it only with your consent,
asked apart from the consent to send, and only with raw data, because the
maintainer regrades every run from it before anything is published. Only an
evaluation of the Full set can enter. Normal has an overall score too, shown on
the result page, but no leaderboard; for Quick, a module's tasks, such as the
German ones, or your own selection of tasks the question is not asked either.

After `Include raw data? [y/N]` is answered with `y`, the command shows this text,
verbatim, and asks `Publish on the leaderboard? [y/N]`; only `y` agrees:

```text
With raw data, the maintainer can also publish this result on the
Werkbank leaderboard, if you agree.

Published are only the values the leaderboard shows: the score with
its interval, the rates per category, the security line, the censored
attempts, the tasks solved per pass, the recorded setup (model,
quantization, engine and version, hardware statement, hardware class
and memory type, thinking setting, context, per-response cap, sampling
and streaming), the Werkbank version, your display name and the date
of the submission. Never published: the raw data and the installation
ID.

An entry is published only after the maintainer has regraded every run
from its raw data and the regrade matches what you sent; otherwise the
submission stays an ordinary one and nothing is published. A display
name is optional; without one the entry shows "anonymous". A name that
insults or impersonates someone is replaced by "anonymous".

To withdraw, write to the deletion contact shown above and quote your
installation ID: from the leaderboard alone, or everything, which also
deletes your submissions. A withdrawal takes effect within 30 days. A
published entry stays until you withdraw it, also after its submission
and raw data are deleted at the end of their 12 months.
```

After a `y` it asks `Display name (Enter for anonymous):`. The name is optional
and may be a pseudonym; it belongs to no account, so others may use the same one.
Spaces at either end are removed and the name is normalized (Unicode NFC). It has
1 to 32 characters: letters (umlauts and other letters included), digits 0-9,
spaces and `-`, `_` and `.`, and it starts and ends with a letter or digit. So it
can never be an e-mail address or a URL. Another answer is asked again with this
message:

```text
Use 1 to 32 letters, digits, spaces, -, _ or ., starting and ending with a letter or digit, or press Enter for anonymous.
```

The [privacy check](#privacy-check) looks for this computer's own values in the
name too, such as your user name or host name. A finding names its kind, never the
value, and the name is asked again:

```text
The privacy check found <kind> in the name; choose another name, or press Enter for anonymous.
```

The envelope records the answer in `leaderboard` and the text shown in
`consent.leaderboard` ([the envelope](#the-envelope)). To withdraw an entry, write
to the deletion contact and quote your installation ID, as the text above says:
from the leaderboard alone, which keeps your submissions, or everything, which
deletes them as well.

## Privacy check

Before the summary, the command searches the exact bytes of the archive's files,
the files inside each run's export included, for data that identifies you or your
computer. On the result page it reads the text, not the markup or the stylesheet.

**In what is always sent** it looks for these patterns. A path, drive letter,
network share, address or host with a port counts only at the start of a token,
that is, not right after a letter, a digit, `-`, `.` or `/`.

- a URL (`scheme://`);
- an absolute local path: one that starts with `/home/`, `/Users/`, `/root/`,
  `/mnt/`, `/media/`, `/srv/`, `/opt/`, `/var/`, `/tmp/`, `/run/`, `/Volumes/` or
  `~/`, a drive letter such as `C:\` or `C:/`, a network share `\\host\`, or
  `file:`;
- the word `localhost`;
- an IPv4 address in a loopback, private, link-local or shared range (127.0.0.0/8,
  10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 169.254.0.0/16, 100.64.0.0/10);
- a host name followed by a port, such as `gpu-box:8000`: a name that starts with
  a letter and holds only letters, digits, `-` and `.`, then `:` and a port from 1
  to 65535.

Model names such as `org/model` and relative names such as
`dataset/data/pools.csv` do not match.

**Everywhere, raw data included,** it looks for this computer's own values, as
whole words and ignoring case: the host name, short and full, your user name, the
home directory, the working directory, the absolute paths of the submitted
directories and, when `werkbank start` or the browser interface runs the
submission, the endpoint address and its host. It skips values shorter than three
characters, the generic names `localhost`, `ubuntu`, `debian`, `fedora`,
`raspberrypi`, `root`, `user`, `admin` and `runner`, and values equal to a word
that Werkbank writes into every submission itself, such as a heading, a category
name, a task ID or the engine names `vllm`, `llama.cpp`, `sglang`, `ollama` and
`lmstudio`. Task files
identical to this build's own are skipped as well.

While the environment variable `WERKBANK_API_KEY` is set, its value is looked for
as well and reported as `an API key`: anywhere in a line, with or without a word
around it, as it stands or with its characters escaped, and even where it is a
word Werkbank writes itself. A key of fewer than eight characters, a placeholder
such as `EMPTY` or `none`, is not looked for. Should an ordinary longer word that
serves as a key stop a submission, unset the variable for `werkbank submit`, as
submitting needs no key.

**In the raw data** a run whose recorded container image names a registry host or
a namespace, such as `ghcr.io/<user>/python@sha256:...`, is a finding, because the
name stands in the run's records and task files. A bare name such as
`python@sha256:...` is none, and neither is this build's default image,
`ghcr.io/aisyndicate-ch/werkbank-runtime` at the digest this build pins: every
tester's run records that same name. The same name with another digest stays a
finding.

What a finding does:

- **In what is always sent:** nothing is sent. The command names the file, the
  line and the kind of finding, never the matched value. Fix the source, for
  example a model name, a hardware label or a serving-note value that holds a host
  name, then write the evaluation and the result page again.
- **Only in the raw data:** the raw data is dropped, with the reason. The rest can
  still be confirmed.
- **In the display name:** the name is asked again ([leaderboard](#leaderboard)).

The check lowers the risk. It cannot prove that free text such as a model name or
a hardware label holds no identifying word, which is why the summary shows those
values before you confirm.

## Installation ID

The installation ID lets the maintainer tell submissions apart and delete yours on
request, without an account. It is 16 random bytes written as 32 hexadecimal
digits. The summary shows it, and it is stored the first time you confirm sending
or writing an archive, in `$XDG_CONFIG_HOME/werkbank/installation-id`
(`~/.config/werkbank/installation-id` when the variable is unset or not an
absolute path; directory mode 0700, file mode 0600, one line). Later submissions
reuse it. An unreadable or malformed file stops the submission with a message and
is never replaced.

Deleting the file resets the ID: the next confirmed submission creates a new one,
and earlier submissions stay under the old one. The ID never enters a run, an
evaluation or an export.

## Verifying what is sent

`--write FILE` runs the same steps and, after `Write the file? [y/N]`, writes the
archive to a new local file instead of sending it, so you can inspect every byte:

```sh
.venv/bin/werkbank submit runs/eval-1 --result runs/result-1 --write submission.tar.gz
tar -tzf submission.tar.gz
sha256sum submission.tar.gz
```

The archive is deterministic: the same files, answers, raw-data choice and
leaderboard answers give the same bytes. A later send with the same answers therefore prints, before
`Send now? [y/N]`, the SHA-256 of the file you inspected, and the receiver's
receipt is that SHA-256. The next two sections describe the file.

## The archive

The archive is one tar file compressed with gzip:

```text
submission.json                 the envelope, always the first member
evaluation/...                  every file of the evaluation directory, as written
result/...                      every file of the result page, as written
raw/<run_id>/export.json        only with raw data: the export index of the run
raw/<run_id>/run.tar            only with raw data: the export archive of the run
```

- After `submission.json` the members follow in sorted path order. The archive
  holds regular files only: no directories, links or devices.
- Every member has mode 0644, owner and group 0, empty owner and group names and
  modification time 0. The gzip header has modification time 0 and no file name.
- Limits: the compressed archive is at most 64 MiB (67,108,864 bytes); it has at
  most 10,000 members, whose bytes total at most 2 GiB (2,147,483,648) unpacked.
  The command builds nothing larger. An archive with raw data over 64 MiB is
  offered without raw data.

## The envelope

`submission.json` holds exactly these fields, in this order
([schema](../schemas/submission.schema.json)):

| Field | Content |
| --- | --- |
| `format` | `werkbank-submission-v2` |
| `installation_id` | the installation ID |
| `werkbank` | `version` and `harness_sha256`, the build digest, of the submitting build |
| `evaluation` | `format`, as `evaluation.json` states it, and `sha256`, the digest of `evaluation.json` |
| `case_set` | `name`, the registered case set whose list holds exactly these tasks, or `null`, and `case_ids`, the tasks every pool was evaluated on |
| `runs` | per evaluated run `run_id`, `plan_sha256` and `run_sha256`, as the evaluation records them |
| `answers` | `format` `werkbank-questions-v1` and one field per question with the chosen ID, or `null` when skipped |
| `consent` | `text` `werkbank-consent-v3`, the consent text shown; `raw_data`, `true` when raw data is included; and `leaderboard`, `werkbank-leaderboard-consent-v1`, the leaderboard's consent text, when it was shown, otherwise `null` |
| `leaderboard` | `publish`, `true` only when you agreed to the leaderboard, and `display_name`, your display name, or `null` for anonymous |
| `files` | `path`, `bytes` and `sha256` of every other member, sorted by path |

Its only free text is your display name, and it holds no time stamp. Submissions
of earlier builds have the format `werkbank-submission-v1`, without `leaderboard`
and without `consent.leaderboard`.

## Sending

The command sends one `POST` to the receiver address over HTTPS, checked against
the system's certificates. There is no account, cookie or token. The request has
these headers besides `Host`:

| Header | Value |
| --- | --- |
| `Content-Type` | `application/gzip` |
| `Content-Length` | the archive's size in bytes |
| `X-Werkbank-Submission` | `werkbank-submission-v2` |
| `X-Werkbank-Installation` | the installation ID |
| `X-Werkbank-Content-SHA256` | the archive's SHA-256 |
| `User-Agent` | `werkbank/<version>` |
| `Accept` | `application/json` |

Proxy settings are ignored, a redirect is never followed and the request is never
repeated automatically. Connecting and each read or write time out after
60 seconds, and at most 16 KiB of the answer is read. Error messages never contain
the address, a header or the archive.

## Receiver answers

| Answer | The command shows |
| --- | --- |
| 201, stored, with the archive's SHA-256 as the receipt | "Received, receipt" and the receipt's first 12 digits |
| 200, an identical archive is already stored | "Already received; nothing new was stored." |
| 400 | "Refused as malformed; nothing was stored. Please report this." |
| 413 | "Too large for the receiver; nothing was stored.", and with raw data included "Send it again without raw data." |
| 415 | "The receiver does not accept this format; update Werkbank." |
| 429 | "Too many submissions; try again in N minutes.", N rounded up and at least 1 ("1 minute" for N = 1), or "try again later" when the answer gives no waiting time in whole seconds |
| 5xx | "Receiver unavailable; nothing was retried. Try again later." |
| 3xx | "The receiver redirected; the redirect was not followed. Nothing was confirmed." |
| any other status | "Unexpected answer (HTTP nnn); nothing was confirmed." |

Only a 201 or 200 answer whose receipt is the archive's own SHA-256 confirms a
submission. No answer, a timeout or a 201 or 200 answer without such a receipt
shows "No confirmed receipt; the submission may have arrived." Sending the same
archive again is safe: the receiver stores an identical archive once.

## Retention and deletion

- Only the maintainer has access to stored submissions and uses them to improve
  Werkbank and its tasks. Nothing is published without the sender's agreement.
- A submission is kept until you ask for its deletion, at most 12 months after it
  arrives.
- To delete, write to the deletion contact shown in the summary and quote your
  installation ID. Every submission under it is deleted, and it is gone from the
  receiver's backups within 30 days.
- An entry on the leaderboard is withdrawn the same way, quoting your installation
  ID: from the leaderboard alone, or completely. A withdrawal takes effect within
  30 days. A published entry, its published values alone, stays until you withdraw
  it, also after its submission is deleted.
- Your IP address is never stored with a submission. The receiver's server logs
  keep it for at most 7 days, only to limit abuse.
